"""Utilities for managing TIDAL OAuth access tokens."""
from __future__ import annotations

import json
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import requests


class TidalOAuthError(RuntimeError):
    """Base error raised when an OAuth interaction fails."""


class TidalAuthorizationRequired(TidalOAuthError):
    """Raised when no valid authorization is available."""


@dataclass
class _TokenData:
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[float] = None

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> "_TokenData":
        return cls(
            access_token=mapping.get("access_token"),
            refresh_token=mapping.get("refresh_token"),
            expires_at=mapping.get("expires_at"),
        )

    def to_mapping(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.access_token:
            payload["access_token"] = self.access_token
        if self.refresh_token:
            payload["refresh_token"] = self.refresh_token
        if self.expires_at:
            payload["expires_at"] = self.expires_at
        return payload


class TidalOAuthClient:
    """Encapsulates the logic for fetching and refreshing TIDAL OAuth tokens."""

    AUTHORIZATION_URL = "https://login.tidal.com/authorize"
    TOKEN_URL = "https://auth.tidal.com/v1/oauth2/token"
    DEFAULT_SCOPE = "user.read playlists.read"
    _EXPIRY_MARGIN = 120  # seconds before expiry to refresh the token

    def __init__(
        self,
        *,
        client_id: Optional[str],
        client_secret: Optional[str],
        redirect_uri: Optional[str],
        token_store_path: Path,
        scope: Optional[str] = None,
        http_client=requests,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scope = scope or self.DEFAULT_SCOPE
        self._http = http_client
        self._token_store_path = Path(token_store_path)
        self._token_data = self._load_tokens()

    # ------------------------------------------------------------------
    # Persistent storage helpers
    # ------------------------------------------------------------------
    def _load_tokens(self) -> _TokenData:
        try:
            with self._token_store_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except FileNotFoundError:
            return _TokenData()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise TidalOAuthError(
                f"Failed to parse token cache at {self._token_store_path}: {exc}"
            ) from exc

        return _TokenData.from_mapping(payload)

    def _save_tokens(self) -> None:
        payload = self._token_data.to_mapping()
        self._token_store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._token_store_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp_path.replace(self._token_store_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_access_token(self, force_refresh: bool = False) -> str:
        """Return a valid access token, refreshing it if required."""

        if force_refresh:
            self._token_data.access_token = None
            self._token_data.expires_at = None

        if self._token_data.access_token and self._token_data.expires_at:
            if (self._token_data.expires_at - self._EXPIRY_MARGIN) > time.time():
                return self._token_data.access_token

        if self._token_data.refresh_token:
            refreshed = self._refresh_access_token()
            if refreshed:
                return refreshed

        raise TidalAuthorizationRequired(
            "No valid TIDAL access token is available. Run with --authorize-tidal "
            "to sign in and grant access."
        )

    def revoke_cached_access_token(self) -> None:
        """Discard the cached access token so that the next call refreshes it."""

        self._token_data.access_token = None
        self._token_data.expires_at = None
        self._save_tokens()

    # ------------------------------------------------------------------
    # Authorization / token exchange helpers
    # ------------------------------------------------------------------
    def _refresh_access_token(self) -> Optional[str]:
        if not self._token_data.refresh_token:
            return None

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._token_data.refresh_token,
        }
        if self.client_id:
            data["client_id"] = self.client_id
        if self.client_secret:
            data["client_secret"] = self.client_secret

        response = self._http.post(
            self.TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        self._update_tokens_from_response(payload)
        return self._token_data.access_token

    def authorize_interactively(self, authorization_response_url: Optional[str] = None) -> str:
        """Kick off a browser-based authorization flow.

        Returns the access token retrieved during the flow.
        """

        if not self.client_id or not self.client_secret or not self.redirect_uri:
            raise TidalOAuthError(
                "TIDAL OAuth credentials are incomplete. Set tidal.client_id, "
                "tidal.client_secret, and tidal.redirect_uri in config.yml."
            )

        state = secrets.token_urlsafe(16)
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": self.scope,
            "state": state,
        }
        authorization_url = f"{self.AUTHORIZATION_URL}?{urlencode(params)}"

        if authorization_response_url is None:
            message_lines = [
                "To authorize Plex Music Builder to access your TIDAL account:",
                f"  1. Open the following URL in a browser: {authorization_url}",
                "  2. After approving access, copy the full redirect URL from your browser.",
                "  3. Paste that URL here and press Enter.",
            ]
            print("\n".join(message_lines), file=sys.stderr)
            authorization_response_url = input("Redirect URL: ").strip()
        elif not authorization_response_url:
            raise TidalOAuthError("Empty authorization response URL provided")

        parsed = urlparse(authorization_response_url)
        query_params = parse_qs(parsed.query)
        returned_state = query_params.get("state", [None])[0]
        if returned_state and returned_state != state:
            raise TidalOAuthError(
                "State returned by TIDAL does not match the request. Aborting."
            )

        code_values = query_params.get("code")
        if not code_values:
            raise TidalOAuthError("Authorization response did not contain a code parameter")

        code = code_values[0]
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret

        response = self._http.post(
            self.TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        self._update_tokens_from_response(payload)
        return self._token_data.access_token or ""

    def _update_tokens_from_response(self, payload: Dict[str, Any]) -> None:
        access_token = payload.get("access_token")
        if not access_token:
            raise TidalOAuthError("Token response did not include an access_token")

        expires_in = payload.get("expires_in")
        expires_at = None
        if isinstance(expires_in, (int, float)):
            expires_at = time.time() + float(expires_in)

        refresh_token = payload.get("refresh_token") or self._token_data.refresh_token

        self._token_data = _TokenData(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
        )
        self._save_tokens()

    # ------------------------------------------------------------------
    # Helpers for tests and debugging
    # ------------------------------------------------------------------
    def dump_cached_tokens(self) -> Dict[str, Any]:
        """Return the cached token payload (excluding sensitive fields)."""

        return self._token_data.to_mapping()


__all__ = [
    "TidalOAuthClient",
    "TidalOAuthError",
    "TidalAuthorizationRequired",
]
