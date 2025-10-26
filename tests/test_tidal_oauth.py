import json
import time
from pathlib import Path

import pytest
import requests

from tidal_auth import TidalAuthorizationRequired, TidalOAuthClient


class _DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._payload


class _DummyHTTPClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, data, headers, timeout):
        if not self._responses:
            raise AssertionError("No more HTTP responses configured")
        payload = self._responses.pop(0)
        self.calls.append((url, data))
        return _DummyResponse(payload)


def _write_tokens(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_get_access_token_from_cache(tmp_path):
    token_path = tmp_path / "tokens.json"
    _write_tokens(
        token_path,
        {
            "access_token": "cached-token",
            "refresh_token": "refresh-token",
            "expires_at": time.time() + 600,
        },
    )

    client = TidalOAuthClient(
        client_id="id",
        client_secret="secret",
        redirect_uri="http://localhost/callback",
        token_store_path=token_path,
    )

    assert client.get_access_token() == "cached-token"


def test_refresh_token_when_expired(tmp_path):
    token_path = tmp_path / "tokens.json"
    _write_tokens(
        token_path,
        {
            "access_token": "old-token",
            "refresh_token": "refresh-token",
            "expires_at": time.time() - 10,
        },
    )

    http_client = _DummyHTTPClient(
        [
            {
                "access_token": "new-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
            }
        ]
    )

    client = TidalOAuthClient(
        client_id="id",
        client_secret="secret",
        redirect_uri="http://localhost/callback",
        token_store_path=token_path,
        http_client=http_client,
    )

    assert client.get_access_token() == "new-token"
    assert http_client.calls
    reloaded = json.loads(token_path.read_text("utf-8"))
    assert reloaded["access_token"] == "new-token"


def test_missing_tokens_raise_authorization_required(tmp_path):
    token_path = tmp_path / "tokens.json"

    client = TidalOAuthClient(
        client_id="id",
        client_secret="secret",
        redirect_uri="http://localhost/callback",
        token_store_path=token_path,
    )

    with pytest.raises(TidalAuthorizationRequired):
        client.get_access_token()
