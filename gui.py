from __future__ import annotations

import re
import json
import os
import math
import mimetypes
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, IO, Iterable, List, Optional, Set, Tuple

import yaml
from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename


def _represent_ordered_dict(dumper: yaml.Dumper, data: OrderedDict) -> Any:
    """Ensure OrderedDict values can be serialized by ``yaml.safe_dump``."""

    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


yaml.SafeDumper.add_representer(OrderedDict, _represent_ordered_dict)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_SEARCH_PATHS: List[Path] = []


def _resolve_config_path() -> Path:
    candidates: List[Path] = []

    env_override = os.environ.get("PMB_CONFIG_PATH")
    if env_override:
        candidates.append(Path(env_override).expanduser())

    candidates.append(Path("/app/config.yml"))
    candidates.append(BASE_DIR / "config.yml")

    CONFIG_SEARCH_PATHS[:] = candidates

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _resolve_runtime_dir(config_dir: Path) -> Path:
    env_override = os.environ.get("PMB_RUNTIME_DIR")
    if env_override:
        return Path(env_override).expanduser()

    app_dir = Path("/app")
    if app_dir.exists():
        return app_dir

    return config_dir


CONFIG_PATH = _resolve_config_path()
CONFIG_DIR = CONFIG_PATH.parent
RUNTIME_DIR = _resolve_runtime_dir(CONFIG_DIR).resolve()

PLAYLISTS_PATH = CONFIG_DIR / "playlists.yml"
LEGEND_PATH = CONFIG_DIR / "legend.txt"
if not LEGEND_PATH.exists():
    LEGEND_PATH = BASE_DIR / "legend.txt"

DEFAULT_ALLMUSIC_CACHE = (RUNTIME_DIR / "allmusic_popularity.json").resolve()
DEFAULT_LOG_PATH = (RUNTIME_DIR / "logs/plex_music_builder.log").resolve()

_CONFIG_CACHE: Dict[str, Any] = {"mtime": None, "data": {}}


def _resolve_path_setting(raw_value: Any, default_path: Path) -> Path:
    if isinstance(raw_value, str) and raw_value.strip():
        candidate = Path(raw_value.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = (CONFIG_DIR / candidate).resolve()
        return candidate
    return default_path


@dataclass
class _ProcessHandle:
    process: subprocess.Popen
    job: Dict[str, Any]
    log_thread: Optional[threading.Thread] = None
    stop_requested: bool = False
    started_at: datetime = field(default_factory=lambda: _utcnow())
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    last_message: Optional[str] = None


def _load_config_data() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}

    try:
        stat_result = CONFIG_PATH.stat()
    except OSError:
        return {}

    cached_mtime = _CONFIG_CACHE.get("mtime")
    if cached_mtime == getattr(stat_result, "st_mtime", None):
        cached_data = _CONFIG_CACHE.get("data")
        return cached_data if isinstance(cached_data, dict) else {}

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config_data = yaml.safe_load(config_file) or {}
    except Exception:
        config_data = {}

    _CONFIG_CACHE["mtime"] = getattr(stat_result, "st_mtime", None)
    _CONFIG_CACHE["data"] = config_data
    return config_data if isinstance(config_data, dict) else {}


CONFIG_LOG_LEVEL_OPTIONS: List[Dict[str, str]] = [
    {"value": "DEBUG", "label": "Debug"},
    {"value": "INFO", "label": "Info"},
    {"value": "WARNING", "label": "Warning"},
    {"value": "ERROR", "label": "Error"},
    {"value": "CRITICAL", "label": "Critical"},
]

CONFIG_FORM_DEFINITION: "OrderedDict[str, Dict[str, Any]]" = OrderedDict(
    [
        (
            "plex",
            {
                "label": "Plex Server",
                "description": (
                    "Configure how the builder connects to your Plex server. "
                    "These values mirror the entries in config.yml."
                ),
                "fields": [
                    {
                        "key": "PLEX_URL",
                        "label": "Server URL",
                        "type": "string",
                        "input": "text",
                        "placeholder": "http://localhost:32400",
                        "helper": "Include the protocol (http or https).",
                    },
                    {
                        "key": "PLEX_TOKEN",
                        "label": "Access Token",
                        "type": "string",
                        "input": "password",
                        "sensitive": True,
                        "helper": "Stored as plain text in config.yml.",
                    },
                    {
                        "key": "library_name",
                        "label": "Music Library Name",
                        "type": "string",
                        "input": "text",
                        "placeholder": "Music",
                    },
                ],
            },
        ),
        (
            "logging",
            {
                "label": "Logging",
                "description": (
                    "Control the verbosity and location of builder log output."
                ),
                "fields": [
                    {
                        "key": "level",
                        "label": "Log Level",
                        "type": "select",
                        "input": "select",
                        "options": CONFIG_LOG_LEVEL_OPTIONS,
                        "helper": "Choose the minimum log severity to capture.",
                    },
                    {
                        "key": "file",
                        "label": "Log File",
                        "type": "string",
                        "input": "text",
                        "placeholder": str(DEFAULT_LOG_PATH),
                        "helper": "Relative paths are resolved from the config directory.",
                    },
                    {
                        "key": "playlist_debug_dir",
                        "label": "Playlist Debug Directory",
                        "type": "string",
                        "input": "text",
                        "helper": "Optional folder for individual playlist debug dumps.",
                    },
                ],
            },
        ),
        (
            "runtime",
            {
                "label": "Runtime Behaviour",
                "description": (
                    "Tune how the builder schedules work and interacts with Plex."
                ),
                "fields": [
                    {
                        "key": "run_forever",
                        "label": "Run Continuously",
                        "type": "boolean",
                        "input": "checkbox",
                        "helper": "Repeat builds on a schedule instead of exiting after one pass.",
                    },
                    {
                        "key": "cache_only",
                        "label": "Cache Only Mode",
                        "type": "boolean",
                        "input": "checkbox",
                        "helper": "Skip playlist updates and only refresh cached metadata.",
                    },
                    {
                        "key": "build_all_on_start",
                        "label": "Build All on Start",
                        "type": "boolean",
                        "input": "checkbox",
                        "helper": "Automatically start building playlists when services launch.",
                    },
                    {
                        "key": "refresh_interval_minutes",
                        "label": "Refresh Interval (minutes)",
                        "type": "integer",
                        "input": "number",
                        "min": 1,
                    },
                    {
                        "key": "save_interval",
                        "label": "Save Interval",
                        "type": "integer",
                        "input": "number",
                        "min": 1,
                        "helper": "Cache progress every N processed items.",
                    },
                    {
                        "key": "max_workers",
                        "label": "Max Parallel Playlists",
                        "type": "integer",
                        "input": "number",
                        "min": 1,
                    },
                    {
                        "key": "playlist_chunk_size",
                        "label": "Playlist Chunk Size",
                        "type": "integer",
                        "input": "number",
                        "min": 1,
                        "helper": "How many tracks to batch when syncing to Plex.",
                    },
                ],
            },
        ),
        (
            "allmusic",
            {
                "label": "AllMusic Integration",
                "description": (
                    "Control how popularity data is fetched from AllMusic and cached."
                ),
                "fields": [
                    {
                        "key": "enabled",
                        "label": "Enable AllMusic",
                        "type": "boolean",
                        "input": "checkbox",
                    },
                    {
                        "key": "cache_file",
                        "label": "Cache File",
                        "type": "string",
                        "input": "text",
                        "placeholder": str(DEFAULT_ALLMUSIC_CACHE),
                    },
                    {
                        "key": "timeout",
                        "label": "Request Timeout (seconds)",
                        "type": "integer",
                        "input": "number",
                        "min": 1,
                    },
                    {
                        "key": "google_min_interval",
                        "label": "Google Search Interval (seconds)",
                        "type": "float",
                        "input": "number",
                        "min": 0.0,
                        "step": 0.1,
                    },
                    {
                        "key": "google_backoff_seconds",
                        "label": "Google Backoff (seconds)",
                        "type": "float",
                        "input": "number",
                        "min": 0.0,
                        "step": 0.1,
                    },
                    {
                        "key": "user_agent",
                        "label": "User Agent",
                        "type": "string",
                        "input": "text",
                    },
                ],
            },
        ),
    ]
)

_CONFIG_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_CONFIG_FALSE_VALUES = {"0", "false", "no", "off", "n"}


def _coerce_config_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered:
            return False
        if lowered in _CONFIG_TRUE_VALUES:
            return True
        if lowered in _CONFIG_FALSE_VALUES:
            return False
    raise ValueError("Enter true or false.")


def _coerce_config_number(value: Any, field: Dict[str, Any], *, is_float: bool) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        numeric: float = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            raise ValueError("Enter a valid number.")

    if not math.isfinite(numeric):
        raise ValueError("Enter a valid number.")

    minimum = field.get("min")
    maximum = field.get("max")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"Value must be at least {minimum}.")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"Value must be at most {maximum}.")

    if not is_float:
        return float(int(round(numeric)))
    return numeric


def _coerce_config_string(value: Any, field: Dict[str, Any]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _coerce_config_select(value: Any, field: Dict[str, Any]) -> Optional[str]:
    options = field.get("options") or []
    if value is None:
        return None

    candidate = str(value).strip()
    if not candidate:
        return None

    option_values = {str(option.get("value")) for option in options}
    if candidate in option_values:
        return candidate

    upper_candidate = candidate.upper()
    for option in options:
        option_value = str(option.get("value"))
        if option_value.upper() == upper_candidate:
            return option_value

    valid = ", ".join(option_values)
    raise ValueError(f"Choose one of: {valid}.")


def _normalize_field_display_value(value: Any, field: Dict[str, Any]) -> Any:
    input_type = field.get("input")
    if input_type == "checkbox":
        try:
            return _coerce_config_boolean(value)
        except ValueError:
            return bool(value)

    if value is None:
        default = field.get("default")
        return "" if default is None else str(default)

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value)


def _extract_section_values(config_data: Dict[str, Any], section_id: str) -> Dict[str, Any]:
    if section_id == "plex":
        plex_cfg = config_data.get("plex")
        if isinstance(plex_cfg, dict):
            return plex_cfg.copy()
        extracted: Dict[str, Any] = {}
        for key in ("PLEX_URL", "PLEX_TOKEN", "library_name"):
            if key in config_data:
                extracted[key] = config_data.get(key)
        return extracted

    section = config_data.get(section_id)
    return section.copy() if isinstance(section, dict) else {}


def _apply_section_updates(
    config_data: Dict[str, Any], section_id: str, updates: Dict[str, Any]
) -> Dict[str, Any]:
    updated = dict(config_data)
    section_values = _extract_section_values(updated, section_id)

    for key, value in updates.items():
        if value is None:
            section_values.pop(key, None)
        else:
            section_values[key] = value

    if section_id == "plex":
        if section_values:
            updated["plex"] = section_values
        else:
            updated.pop("plex", None)
        for legacy_key in ("PLEX_URL", "PLEX_TOKEN", "library_name"):
            if legacy_key in updates:
                legacy_value = updates[legacy_key]
                if legacy_value is None:
                    updated.pop(legacy_key, None)
                else:
                    updated[legacy_key] = legacy_value
        return updated

    if section_values:
        updated[section_id] = section_values
    else:
        updated.pop(section_id, None)
    return updated


def _coerce_config_field_value(value: Any, field: Dict[str, Any]) -> Optional[Any]:
    field_type = field.get("type")
    if field_type == "boolean":
        return _coerce_config_boolean(value)
    if field_type == "integer":
        coerced = _coerce_config_number(value, field, is_float=False)
        return None if coerced is None else int(coerced)
    if field_type == "float":
        coerced = _coerce_config_number(value, field, is_float=True)
        return None if coerced is None else float(coerced)
    if field_type == "select":
        return _coerce_config_select(value, field)
    return _coerce_config_string(value, field)


def _build_config_sections_payload(
    config_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if config_data is None:
        config_data = _load_config_data()

    sections_payload: List[Dict[str, Any]] = []
    for section_id, definition in CONFIG_FORM_DEFINITION.items():
        values = _extract_section_values(config_data, section_id)
        fields_payload: List[Dict[str, Any]] = []
        for field in definition.get("fields", []):
            key = field.get("key")
            if not key:
                continue
            raw_value = values.get(key, field.get("default"))
            display_value = _normalize_field_display_value(raw_value, field)
            field_payload: Dict[str, Any] = {
                "key": key,
                "label": field.get("label", key),
                "input": field.get("input", "text"),
                "type": field.get("type", "string"),
                "value": display_value,
            }
            for attr in ("placeholder", "helper", "step", "min", "max"):
                if field.get(attr) is not None:
                    field_payload[attr] = field.get(attr)
            if field.get("options"):
                field_payload["options"] = field.get("options")
            if field.get("sensitive"):
                field_payload["sensitive"] = True
            fields_payload.append(field_payload)

        section_payload = {
            "id": section_id,
            "label": definition.get("label", section_id.title()),
            "description": definition.get("description"),
            "fields": fields_payload,
        }
        sections_payload.append(section_payload)

    return {
        "sections": sections_payload,
        "path": str(CONFIG_PATH),
        "search_paths": [str(path) for path in CONFIG_SEARCH_PATHS],
    }


def _save_config_data(config_data: Dict[str, Any]) -> Dict[str, Any]:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = deepcopy(config_data)
    with CONFIG_PATH.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(serializable, config_file, sort_keys=False, allow_unicode=True)

    try:
        stat_result = CONFIG_PATH.stat()
        _CONFIG_CACHE["mtime"] = getattr(stat_result, "st_mtime", None)
    except OSError:
        _CONFIG_CACHE["mtime"] = None

    _CONFIG_CACHE["data"] = deepcopy(config_data)
    return config_data


def _schedule_restart(delay: float = 1.0) -> None:
    def _delayed_exit() -> None:
        time.sleep(max(0.0, delay))
        os._exit(0)

    thread = threading.Thread(target=_delayed_exit, name="pmb-restart", daemon=True)
    thread.start()

GENERAL_LOG_RETENTION = timedelta(minutes=5)
GENERAL_LOG_DISPLAY_LIMIT = 3
PLAYLIST_ACTIVITY_RETENTION = timedelta(minutes=15)

# Human-friendly overrides for certain Plex field names
FIELD_LABEL_OVERRIDES: Dict[str, str] = {
    "parentYear": "Album Year",
    "grandparentTitle": "Artist Name",
    "grandparentRatingKey": "Artist Key",
    "grandparentGuid": "Artist GUID",
    "grandparentThumb": "Artist Artwork",
    "grandparentArt": "Artist Background",
    "parentTitle": "Album Title",
    "parentRatingKey": "Album Key",
    "parentGuid": "Album GUID",
    "parentThumb": "Album Cover",
    "parentArt": "Album Background",
    "originallyAvailableAt": "Release Date",
    "parentIndex": "Disc Number",
    "media.parts": "Media Parts",
    "media.file": "Media File",
    "media.container": "Media Container",
    "media.audioCodec": "Audio Codec",
    "media.bitrate": "Media Bitrate",
    "media.audioProfile": "Audio Profile",
    "artist.id": "Artist ID",
    "artist.guid": "Artist GUID",
    "artist.ratingKey": "Artist Rating Key",
    "album.id": "Album ID",
    "album.guid": "Album GUID",
    "album.ratingKey": "Album Rating Key",
    "uuid": "UUID",
    "guid": "GUID",
    "ratingKey": "Rating Key",
    "librarySectionID": "Library Section ID",
}

# Focused subset of Plex metadata fields that are meaningful for
# playlist building. These are exposed in the GUI instead of the
# exhaustive list from ``legend.txt`` to keep the dropdown manageable
# and aligned with the fields the app understands natively.
CURATED_FIELD_CHOICES: List[str] = [
    "artist",
    "album",
    "title",
    "album.year",
    "album.type",
    "album.title",
    "parentIndex",
    "genres",
    "moods",
]

OPERATOR_OPTIONS = OrderedDict(
    [
        ("equals", "Equals"),
        ("does_not_equal", "Does Not Equal"),
        ("contains", "Contains"),
        ("does_not_contain", "Does Not Contain"),
        ("greater_than", "Greater Than"),
        ("less_than", "Less Than"),
    ]
)

SORT_OPTIONS = OrderedDict(
    [
        ("popularity", "Popularity"),
        ("alphabetical", "Alphabetical"),
        ("reverse_alphabetical", "Lacitebahpla"),
        ("oldest_first", "Oldest First"),
        ("newest_first", "Newest First"),
    ]
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _alphabetize_playlist_configs(
    playlists: Iterable[Tuple[str, Dict[str, Any]]]
) -> "OrderedDict[str, Dict[str, Any]]":
    """Return an ``OrderedDict`` sorted alphabetically by playlist title.

    Sorting is case-insensitive to ensure titles like "a" and "A" are
    treated equivalently while preserving the original casing in the
    output.
    """

    return OrderedDict(
        sorted(playlists, key=lambda item: (item[0].casefold(), item[0]))
    )


def _format_timestamp(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


class PlaylistConflictError(Exception):
    """Raised when attempting to save a playlist that already exists."""


class BuildManager:
    _INFO_LINE_PATTERN = re.compile(r"\[(?P<level>[A-Z]+)\]\s*(?P<message>.*)")
    _LOG_MESSAGE_LEVELS = {"INFO", "WARNING", "ERROR", "CRITICAL"}
    _PLAYLIST_QUOTED_PATTERN = re.compile(r"'([^']+)'")
    _FILTERING_PROGRESS_LEGACY_PATTERN = re.compile(
        r"Filtering '(?P<playlist>[^']+)':?\s+"
        r"(?P<percent>\d+(?:\.\d+)?)%[^\d]*"
        r"(?P<current>\d+)\s*/\s*(?P<total>\d+|\?)"
        r"(?:[^\[]*(?P<details>\[[^\]]*\]))?"
    )
    _FILTERING_PROGRESS_CLEAN_PATTERN = re.compile(
        r"Filtering progress for '(?P<playlist>[^']+)':\s*"
        r"(?P<current>\d+)\s*/\s*(?P<total>\d+|\?)"
        r"(?:\s*\((?P<percent>\d+(?:\.\d+)?)%\))?"
        r"(?P<extras>.*)"
    )
    _PROCESSING_PLAYLISTS_PATTERN = re.compile(
        r"Processing\s+(?P<count>\d+)\s+playlist\(s\):\s*(?P<names>.+)",
        re.IGNORECASE,
    )
    def __init__(
        self,
        command: Optional[List[str]] = None,
        work_dir: Optional[Path] = None,
    ) -> None:
        self._command = command or [sys.executable, "main.py"]
        self._work_dir = str(work_dir) if work_dir is not None else None
        self._lock = threading.Lock()
        self._processes: List[_ProcessHandle] = []
        self._last_exit_code: Optional[int] = None
        self._last_message: Optional[str] = None
        self._last_started_at: Optional[datetime] = None
        self._last_finished_at: Optional[datetime] = None
        self._last_all_result: Optional[Dict[str, Any]] = None
        self._last_playlist_results: Dict[str, Dict[str, Any]] = {}
        self._playlist_logs: Dict[str, List[Dict[str, Any]]] = {}
        self._playlist_completed_at: Dict[str, datetime] = {}
        self._general_logs: List[Dict[str, Any]] = []
        self._log_file_path: Optional[Path] = resolve_log_file_path()
        self._log_watcher_thread: Optional[threading.Thread] = None
        self._passive_running = False
        self._passive_job: Optional[Dict[str, Any]] = None
        self._passive_last_started_at: Optional[datetime] = None
        self._passive_last_finished_at: Optional[datetime] = None
        self._passive_last_message: Optional[str] = None
        self._passive_last_state: Optional[str] = None
        self._observed_active_playlists: Set[str] = set()
        self._queued_playlists: List[str] = []
        self._max_initial_log_read = 65536
        self._last_run_state: Optional[str] = None
        self._pending_jobs: List[Dict[str, Any]] = []
        self._start_log_watcher()

    @staticmethod
    def _normalize_playlist_key(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        key = str(name).strip()
        return key or None

    def _find_handle_for_playlist_locked(
        self, playlist_name: Optional[str]
    ) -> Optional[_ProcessHandle]:
        normalized = self._normalize_playlist_key(playlist_name)
        if not normalized:
            return None

        for handle in self._processes:
            job = handle.job or {}
            if job.get("type") == "playlist":
                candidate = self._normalize_playlist_key(job.get("playlist"))
                if candidate == normalized:
                    return handle

        return None

    def _find_handle_for_all_locked(self) -> Optional[_ProcessHandle]:
        for handle in self._processes:
            job = handle.job or {}
            if job.get("type") == "all":
                return handle
        return None

    @classmethod
    def _extract_info_message(cls, line: str) -> Optional[str]:
        raw = line.strip()
        if not raw:
            return None

        match = cls._INFO_LINE_PATTERN.search(raw)
        if match:
            level = match.group("level")
            if level not in cls._LOG_MESSAGE_LEVELS:
                return None
            message = match.group("message").strip()
            return message

        parts = raw.split(":", 1)
        if len(parts) == 2:
            level = parts[0].strip().upper()
            if level in cls._LOG_MESSAGE_LEVELS:
                return parts[1].strip()
        return None

    @classmethod
    def _extract_playlist_from_message(cls, message: str) -> Optional[str]:
        quoted = cls._PLAYLIST_QUOTED_PATTERN.search(message)
        if quoted:
            return cls._normalize_playlist_key(quoted.group(1))

        if message.lower().startswith("building playlist:"):
            _, _, name = message.partition(":")
            return cls._normalize_playlist_key(name)

        return None

    def _append_playlist_log_locked(self, playlist_name: str, message: str) -> None:
        logs = self._playlist_logs.setdefault(playlist_name, [])
        normalized_message = message.strip().lower()
        is_final = (
            normalized_message.startswith("✅")
            or "failed" in normalized_message
            or "stopped" in normalized_message
        )
        entry = {
            "type": "message",
            "text": message,
            "timestamp": _format_timestamp(_utcnow()),
            "is_final": is_final,
        }
        self._playlist_logs[playlist_name] = [entry]
        self._passive_last_message = message
        if not self._processes:
            self._last_message = message

    def _record_filtering_progress_locked(
        self, playlist_name: str, progress: Dict[str, Any]
    ) -> None:
        logs = self._playlist_logs.setdefault(playlist_name, [])
        entry = {
            "type": "progress",
            "text": progress.get("message") or "Filtering tracks…",
            "current": progress.get("current"),
            "total": progress.get("total"),
            "percent": progress.get("percent"),
            "progress": progress.get("progress"),
            "details": progress.get("details"),
            "rate": progress.get("rate"),
            "elapsed": progress.get("elapsed"),
            "eta": progress.get("eta"),
            "timestamp": _format_timestamp(_utcnow()),
        }
        self._playlist_logs[playlist_name] = [entry]
        self._passive_last_message = entry["text"]
        if not self._processes:
            self._last_message = entry["text"]

        handle = self._find_handle_for_playlist_locked(playlist_name)
        if handle is not None:
            progress_snapshot = {
                key: entry.get(key)
                for key in (
                    "current",
                    "total",
                    "percent",
                    "progress",
                    "details",
                    "rate",
                    "elapsed",
                    "eta",
                )
            }
            handle.job["progress"] = progress_snapshot

    def _prune_general_logs_locked(
        self, reference_time: Optional[datetime] = None
    ) -> None:
        if reference_time is None:
            reference_time = _utcnow()

        cutoff = reference_time - GENERAL_LOG_RETENTION
        normalized_entries: List[Dict[str, Any]] = []

        for entry in self._general_logs:
            text: Optional[str]
            timestamp: Optional[datetime]
            normalized_entry: Dict[str, Any]

            if isinstance(entry, dict):
                normalized_entry = dict(entry)
                text = normalized_entry.get("text")
                timestamp = normalized_entry.get("timestamp")
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                    except ValueError:
                        timestamp = None
            elif isinstance(entry, str):
                normalized_entry = {"text": entry}
                text = entry
                timestamp = None
            else:
                continue

            if not text:
                continue

            if not isinstance(timestamp, datetime):
                timestamp = reference_time

            if timestamp < cutoff:
                continue

            normalized_entry["text"] = text
            normalized_entry["timestamp"] = timestamp

            normalized_entries.append(normalized_entry)

        if len(normalized_entries) > 400:
            normalized_entries = normalized_entries[-400:]

        self._general_logs = normalized_entries

    def _append_general_log_locked(self, message: str) -> None:
        entry: Dict[str, Any] = {"text": message, "timestamp": _utcnow()}
        self._general_logs.append(entry)
        self._prune_general_logs_locked(entry["timestamp"])
        self._passive_last_message = message
        if not self._processes:
            self._last_message = message

    def _mark_playlist_started_locked(self, playlist_name: str) -> None:
        normalized = self._normalize_playlist_key(playlist_name)
        if not normalized:
            return
        self._playlist_completed_at.pop(normalized, None)

    def _mark_playlist_completed_locked(
        self, playlist_name: str, completed_at: Optional[datetime] = None
    ) -> None:
        normalized = self._normalize_playlist_key(playlist_name)
        if not normalized:
            return
        self._playlist_completed_at[normalized] = completed_at or _utcnow()

    def _prune_playlist_logs_locked(
        self,
        active_playlists: Set[str],
        waiting_playlists: Iterable[str],
        reference_time: Optional[datetime] = None,
    ) -> None:
        if reference_time is None:
            reference_time = _utcnow()

        cutoff = reference_time - PLAYLIST_ACTIVITY_RETENTION

        waiting_set: Set[str] = set()
        for name in waiting_playlists:
            normalized = self._normalize_playlist_key(name)
            if normalized:
                waiting_set.add(normalized)

        for name, completed_at in list(self._playlist_completed_at.items()):
            if not isinstance(completed_at, datetime):
                continue
            if completed_at > cutoff:
                continue
            if name in active_playlists or name in waiting_set:
                continue
            self._playlist_completed_at.pop(name, None)
            self._playlist_logs.pop(name, None)
            self._last_playlist_results.pop(name, None)

        for raw_name, entries in list(self._playlist_logs.items()):
            normalized = self._normalize_playlist_key(raw_name)
            if not normalized:
                self._playlist_logs.pop(raw_name, None)
                continue
            if normalized in active_playlists or normalized in waiting_set:
                continue
            if normalized in self._playlist_completed_at:
                continue

            most_recent: Optional[datetime] = None
            if isinstance(entries, list):
                for entry in entries:
                    timestamp = entry.get("timestamp") if isinstance(entry, dict) else None
                    candidate: Optional[datetime]
                    if isinstance(timestamp, datetime):
                        candidate = timestamp
                    elif isinstance(timestamp, str):
                        try:
                            candidate = datetime.fromisoformat(timestamp)
                            if candidate.tzinfo is None:
                                candidate = candidate.replace(tzinfo=timezone.utc)
                        except ValueError:
                            candidate = None
                    else:
                        candidate = None

                    if candidate is None:
                        candidate = reference_time

                    if most_recent is None or candidate > most_recent:
                        most_recent = candidate

            if most_recent is None:
                most_recent = reference_time

            if most_recent < cutoff:
                self._playlist_logs.pop(raw_name, None)
                self._last_playlist_results.pop(normalized, None)

    def _remove_waiting_playlist_locked(self, playlist_name: Optional[str]) -> None:
        normalized = self._normalize_playlist_key(playlist_name)
        if not normalized:
            return
        self._queued_playlists = [
            name
            for name in self._queued_playlists
            if self._normalize_playlist_key(name) != normalized
        ]

    def _sync_waiting_from_pending_locked(self) -> None:
        pending_names: List[str] = []
        for job in self._pending_jobs:
            if job.get("type") == "playlist":
                name = job.get("playlist")
                if isinstance(name, str):
                    normalized = name.strip()
                    if normalized:
                        pending_names.append(normalized)
        if not pending_names:
            return
        existing = {
            self._normalize_playlist_key(name)
            for name in self._queued_playlists
            if isinstance(name, str)
        }
        for name in pending_names:
            normalized = self._normalize_playlist_key(name)
            if normalized and normalized not in existing:
                self._queued_playlists.append(name)
                existing.add(normalized)


    def _start_log_watcher(self) -> None:
        if self._log_file_path is None:
            return
        if self._log_watcher_thread and self._log_watcher_thread.is_alive():
            return

        thread = threading.Thread(
            target=self._watch_log_file,
            name="BuildLogWatcher",
            daemon=True,
        )
        self._log_watcher_thread = thread
        thread.start()

    def _watch_log_file(self) -> None:
        if self._log_file_path is None:
            return

        log_path = Path(self._log_file_path)
        last_inode = None
        offset = 0
        read_from_start = True

        while True:
            try:
                stat_info = log_path.stat()
                inode = getattr(stat_info, "st_ino", None)
                if inode != last_inode:
                    last_inode = inode
                    offset = 0
                    read_from_start = True

                with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                    bootstrap = read_from_start
                    if read_from_start:
                        size = stat_info.st_size
                        if size > self._max_initial_log_read:
                            handle.seek(size - self._max_initial_log_read)
                            handle.readline()
                        offset = handle.tell()
                        read_from_start = False
                    else:
                        handle.seek(offset)

                    while True:
                        line = handle.readline()
                        if not line:
                            offset = handle.tell()
                            bootstrap = False
                            time.sleep(0.5)
                            try:
                                current_stat = log_path.stat()
                            except OSError:
                                read_from_start = True
                                offset = 0
                                break
                            if current_stat.st_size < offset:
                                read_from_start = True
                                offset = 0
                                break
                            continue

                        sanitized = line.strip("\r\n")
                        if sanitized:
                            self._handle_log_line(sanitized, is_bootstrap=bootstrap)
            except FileNotFoundError:
                offset = 0
                read_from_start = True
                last_inode = None
                time.sleep(1.0)
            except Exception:
                time.sleep(1.0)

    def _update_passive_state_from_message_locked(
        self, playlist_name: Optional[str], message: str
    ) -> None:
        normalized = message.strip().lower()
        now = _utcnow()

        if playlist_name:
            if normalized.startswith("building playlist:"):
                self._mark_playlist_started_locked(playlist_name)
                self._observed_active_playlists.add(playlist_name)
                if not self._processes and not self._passive_running:
                    self._passive_last_started_at = now
                if not self._processes:
                    self._passive_running = True
                    self._passive_last_state = "running"
                    if not self._passive_job or self._passive_job.get("type") != "all":
                        self._passive_job = {
                            "type": "playlist",
                            "playlist": playlist_name,
                        }
            elif normalized.startswith("build started for playlist"):
                self._mark_playlist_started_locked(playlist_name)
                self._observed_active_playlists.add(playlist_name)
                if not self._processes and not self._passive_running:
                    self._passive_last_started_at = now
                if not self._processes:
                    self._passive_running = True
                    self._passive_last_state = "running"
                    if not self._passive_job or self._passive_job.get("type") != "all":
                        self._passive_job = {
                            "type": "playlist",
                            "playlist": playlist_name,
                        }
            elif (
                normalized.startswith("✅ finished building")
                or "failed" in normalized
                or "stopped" in normalized
            ):
                self._mark_playlist_completed_locked(playlist_name, now)
                if "failed" in normalized:
                    self._passive_last_state = "error"
                elif "stopped" in normalized:
                    self._passive_last_state = "stopped"
                else:
                    self._passive_last_state = "success"
                self._observed_active_playlists.discard(playlist_name)
                if (
                    not self._processes
                    and not self._observed_active_playlists
                    and (not self._passive_job or self._passive_job.get("type") != "all")
                ):
                    self._passive_running = False
                    self._passive_last_finished_at = now
                    self._passive_job = None
        else:
            if normalized.startswith("processing") and "playlist" in normalized:
                if not self._processes and not self._passive_running:
                    self._passive_last_started_at = now
                if not self._processes:
                    self._passive_running = True
                    self._passive_last_state = "running"
                    self._passive_job = {"type": "all"}
            elif normalized.startswith("build for all playlists started"):
                if not self._processes and not self._passive_running:
                    self._passive_last_started_at = now
                if not self._processes:
                    self._passive_running = True
                    self._passive_last_state = "running"
                    self._passive_job = {"type": "all"}
            elif normalized.startswith("✅ all playlists processed") or normalized.startswith(
                "✅ selected playlists processed"
            ):
                if not self._processes:
                    self._passive_running = False
                    self._passive_last_finished_at = now
                    self._passive_job = None
                    self._passive_last_state = "success"
                    self._queued_playlists = []
            elif normalized.startswith("sleeping for") or "completed successfully" in normalized:
                if not self._processes:
                    self._passive_running = False
                    self._passive_last_finished_at = now
                    self._passive_job = None
                    if "completed successfully" in normalized:
                        self._passive_last_state = "success"
                    elif "stopped" in normalized:
                        self._passive_last_state = "stopped"
                    else:
                        self._passive_last_state = "success"
            elif "failed" in normalized:
                if not self._processes:
                    self._passive_last_state = "error"
                    self._passive_running = False
                    self._passive_last_finished_at = now
                    self._passive_job = None
                    self._queued_playlists = []
            elif "stopped" in normalized:
                if not self._processes:
                    self._passive_last_state = "stopped"
                    self._passive_running = False
                    self._passive_last_finished_at = now
                    self._passive_job = None
                    self._queued_playlists = []

    @classmethod
    def _split_processing_playlist_names(
        cls, names_segment: str, expected_count: Optional[int]
    ) -> List[str]:
        segment = (names_segment or "").strip()
        if not segment:
            return []

        if not expected_count or expected_count <= 1:
            normalized = cls._normalize_playlist_key(segment)
            return [normalized] if normalized else []

        parts = segment.split(",")
        names: List[str] = []
        current_parts: List[str] = []
        total_parts = len(parts)

        for index, part in enumerate(parts):
            current_parts.append(part)
            remaining_parts = total_parts - index - 1
            remaining_slots = expected_count - len(names) - 1
            if remaining_slots < 0:
                continue
            if remaining_parts == remaining_slots:
                candidate = ",".join(current_parts).strip()
                normalized = cls._normalize_playlist_key(candidate)
                if normalized:
                    names.append(normalized)
                current_parts = []

        if current_parts:
            candidate = ",".join(current_parts).strip()
            normalized = cls._normalize_playlist_key(candidate)
            if normalized:
                names.append(normalized)

        seen: Set[str] = set()
        ordered_names: List[str] = []
        for name in names:
            if name not in seen:
                seen.add(name)
                ordered_names.append(name)

        return ordered_names

    @classmethod
    def _parse_processing_playlist_message(cls, message: str) -> Optional[List[str]]:
        match = cls._PROCESSING_PLAYLISTS_PATTERN.search(message)
        if not match:
            return None

        names_segment = match.group("names") or ""
        count_text = match.group("count")
        expected_count: Optional[int]
        try:
            expected_count = int(count_text)
        except (TypeError, ValueError):
            expected_count = None

        names = cls._split_processing_playlist_names(names_segment, expected_count)
        return names

    def _update_waiting_playlists_locked(
        self, message: str, playlist_name: Optional[str]
    ) -> None:
        queue_names = self._parse_processing_playlist_message(message)
        if queue_names is not None:
            self._queued_playlists = list(queue_names)
            self._sync_waiting_from_pending_locked()
            return

        normalized_message = message.strip().lower()
        normalized_name = self._normalize_playlist_key(playlist_name)

        if normalized_name and self._queued_playlists:
            if normalized_message.startswith("building playlist:") or normalized_message.startswith(
                "build started for playlist"
            ):
                self._remove_waiting_playlist_locked(playlist_name)
            elif normalized_message.startswith("✅") or "failed" in normalized_message or "stopped" in normalized_message:
                self._remove_waiting_playlist_locked(playlist_name)
        elif normalized_message.startswith("✅ all playlists processed") or normalized_message.startswith(
            "✅ selected playlists processed"
        ) or normalized_message.startswith("sleeping for"):
            self._queued_playlists = []

        self._sync_waiting_from_pending_locked()

    def _build_passive_job_snapshot_locked(self) -> Optional[Dict[str, Any]]:
        if self._passive_job:
            job = self._passive_job.copy()
        elif self._observed_active_playlists:
            playlist_name = sorted(self._observed_active_playlists)[0]
            job = {"type": "playlist", "playlist": playlist_name}
        else:
            return None

        playlist_name = job.get("playlist")
        if playlist_name:
            entries = self._playlist_logs.get(playlist_name) or []
            for entry in reversed(entries):
                if isinstance(entry, dict) and entry.get("type") == "progress":
                    job["progress"] = {
                        key: entry.get(key)
                        for key in (
                            "current",
                            "total",
                            "percent",
                            "progress",
                            "details",
                            "rate",
                            "elapsed",
                            "eta",
                        )
                    }
                    break

        if self._passive_last_started_at and "started_at" not in job:
            job["started_at"] = _format_timestamp(self._passive_last_started_at)

        active_names: Set[str] = set()
        existing_active = job.get("active_playlists")
        if isinstance(existing_active, list):
            for name in existing_active:
                if isinstance(name, str):
                    normalized = name.strip()
                    if normalized:
                        active_names.add(normalized)

        if isinstance(playlist_name, str):
            normalized_playlist = playlist_name.strip()
            if normalized_playlist:
                active_names.add(normalized_playlist)

        for observed in self._observed_active_playlists:
            normalized_observed = str(observed).strip()
            if normalized_observed:
                active_names.add(normalized_observed)

        if active_names:
            job["active_playlists"] = sorted(active_names)
        elif "active_playlists" in job:
            job.pop("active_playlists", None)

        return job

    @classmethod
    def _parse_filtering_progress_line(cls, line: str) -> Optional[Dict[str, Any]]:
        clean_match = cls._FILTERING_PROGRESS_CLEAN_PATTERN.search(line)
        legacy_match = None if clean_match else cls._FILTERING_PROGRESS_LEGACY_PATTERN.search(line)
        if not clean_match and not legacy_match:
            return None

        if clean_match:
            playlist = cls._normalize_playlist_key(clean_match.group("playlist"))
            if not playlist:
                return None

            current_raw = clean_match.group("current")
            total_raw = clean_match.group("total")
            percent_raw = clean_match.group("percent")
            extras_raw = clean_match.group("extras") or ""
        else:
            playlist = cls._normalize_playlist_key(legacy_match.group("playlist"))
            if not playlist:
                return None

            current_raw = legacy_match.group("current")
            total_raw = legacy_match.group("total")
            percent_raw = legacy_match.group("percent")
            details_raw = legacy_match.group("details") or ""
            inner = details_raw.strip().strip("[]")
            extras_raw = ""
            if inner:
                segments = [segment.strip() for segment in inner.split(",") if segment.strip()]
                if segments:
                    extras_raw = " – ".join(segments)

        try:
            current_value = int(current_raw)
        except (TypeError, ValueError):
            current_value = 0

        total_value: Optional[int]
        try:
            total_value = int(total_raw) if total_raw not in {None, "?"} else None
        except (TypeError, ValueError):
            total_value = None

        percent_value: Optional[float]
        try:
            percent_value = float(percent_raw) if percent_raw is not None else None
        except (TypeError, ValueError):
            percent_value = None

        if percent_value is None and total_value:
            percent_value = (float(current_value) / float(total_value)) * 100.0

        progress_fraction: Optional[float] = None
        if percent_value is not None:
            progress_fraction = max(0.0, min(percent_value / 100.0, 1.0))

        elapsed_text: Optional[str] = None
        eta_text: Optional[str] = None
        rate_text: Optional[str] = None
        extra_segments: List[str] = []

        normalized_extras = extras_raw.replace("—", "–")
        segments = [
            segment.strip()
            for segment in re.split(r"\s+[–-]\s+", normalized_extras)
            if segment.strip()
        ]
        for segment in segments:
            lower = segment.lower()
            if lower.startswith("elapsed"):
                _, _, value = segment.partition(" ")
                elapsed_text = value.strip() or segment.strip()
            elif lower.startswith("eta"):
                _, _, value = segment.partition(" ")
                eta_text = value.strip() or segment.strip()
            elif "track/s" in lower:
                cleaned = segment.replace("track/s", " track/s").replace("  ", " ")
                rate_text = cleaned.strip()
            else:
                extra_segments.append(segment)

        total_text = str(total_value) if total_value is not None else "?"
        percent_text = f"{percent_value:.0f}%" if percent_value is not None else None

        details_parts: List[str] = []
        if elapsed_text:
            details_parts.append(f"{elapsed_text} elapsed")
        if eta_text:
            details_parts.append(f"ETA {eta_text}")
        if rate_text:
            details_parts.append(rate_text)
        details_parts.extend(extra_segments)

        details_text = " · ".join(details_parts)
        message = f"Filtering tracks… {current_value}/{total_text}"
        if percent_text:
            message = f"{message} ({percent_text})"
        if details_text:
            message = f"{message} — {details_text}"

        return {
            "playlist": playlist,
            "percent": percent_value,
            "progress": progress_fraction,
            "current": current_value,
            "total": total_value,
            "details": details_text or None,
            "rate": rate_text,
            "elapsed": elapsed_text,
            "eta": eta_text,
            "message": message,
        }

    def _handle_log_line(self, line: str, *, is_bootstrap: bool = False) -> None:
        progress = self._parse_filtering_progress_line(line)
        progress_playlist: Optional[str] = None
        handled_progress = False
        if progress is not None:
            progress_playlist = progress.get("playlist")
            if progress_playlist:
                if is_bootstrap:
                    handled_progress = True
                else:
                    with self._lock:
                        self._record_filtering_progress_locked(
                            progress_playlist, progress
                        )
                        self._observed_active_playlists.add(progress_playlist)
                    handled_progress = True

        info_message = self._extract_info_message(line)
        if info_message is None:
            return

        playlist_name = self._extract_playlist_from_message(info_message)
        lowercase_message = info_message.lower()
        with self._lock:
            if is_bootstrap:
                self._passive_last_message = info_message
                if not self._processes:
                    self._last_message = info_message
            else:
                self._update_passive_state_from_message_locked(
                    playlist_name, info_message
                )
                self._update_waiting_playlists_locked(info_message, playlist_name)
            if playlist_name:
                if handled_progress and playlist_name == progress_playlist:
                    return
                is_start_message = lowercase_message.startswith(
                    "building playlist:"
                ) or lowercase_message.startswith("build started for playlist")
                if not is_bootstrap and lowercase_message.startswith(
                    "building playlist:"
                ):
                    self._playlist_logs[playlist_name] = []
                if not is_bootstrap and is_start_message:
                    self._mark_playlist_started_locked(playlist_name)
                if is_bootstrap:
                    if not self._processes:
                        self._last_message = info_message
                        self._passive_last_message = info_message
                else:
                    self._append_playlist_log_locked(playlist_name, info_message)
                    if (
                        lowercase_message.startswith("✅ finished building")
                        or "failed" in lowercase_message
                        or "stopped" in lowercase_message
                    ):
                        self._mark_playlist_completed_locked(playlist_name)
            elif not handled_progress and not is_bootstrap:
                self._append_general_log_locked(info_message)

    def _consume_process_output(
        self, handle: _ProcessHandle, stream: IO[str]
    ) -> None:
        pending = ""
        try:
            for raw_line in stream:
                if not raw_line:
                    continue
                pending += raw_line
                segments = re.split(r"[\r\n]+", pending)
                if pending and pending[-1] not in {"\r", "\n"}:
                    if segments:
                        pending = segments.pop()
                    else:
                        pending = ""
                else:
                    pending = ""
                for segment in segments:
                    sanitized = segment.strip("\r\n")
                    if sanitized:
                        self._handle_log_line(sanitized)
            if pending:
                sanitized = pending.strip("\r\n")
                if sanitized:
                    self._handle_log_line(sanitized)
        finally:
            try:
                stream.close()
            except Exception:  # pragma: no cover - defensive
                pass
            with self._lock:
                if handle in self._processes:
                    handle.log_thread = None
    def _parallel_limit_locked(self) -> int:
        config_data = _load_config_data()
        runtime_cfg = config_data.get("runtime") or {}
        value = runtime_cfg.get("max_workers")
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = 1
        if numeric < 1:
            numeric = 1
        return numeric

    def _launch_job_locked(self, job: Dict[str, Any]) -> Tuple[bool, str]:
        command = list(self._command)
        job_type = job.get("type")
        playlist_name: Optional[str] = None

        if job_type == "playlist":
            playlist_name = job.get("playlist")
            if not isinstance(playlist_name, str) or not playlist_name.strip():
                message = "Playlist name is required to start a build."
                self._last_message = message
                return False, message
            playlist_name = playlist_name.strip()
            command = command + ["--playlist", playlist_name]
            job_payload: Dict[str, Any] = {"type": "playlist", "playlist": playlist_name}
            job_payload["active_playlists"] = [playlist_name]
            job_message = f"Build started for playlist '{playlist_name}'."
            self._playlist_logs.pop(playlist_name, None)
            self._mark_playlist_started_locked(playlist_name)
        else:
            job_payload = {"type": "all"}
            job_message = "Build for all playlists started."
            self._playlist_logs = {}
            self._general_logs = []
            self._playlist_completed_at = {}

        try:
            process = subprocess.Popen(
                command,
                cwd=self._work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - defensive
            error_message = f"Unable to start build: {exc}"
            self._last_message = error_message
            self._last_started_at = None
            self._last_finished_at = _utcnow()
            self._last_exit_code = None
            self._last_run_state = "error"
            return False, error_message

        handle = _ProcessHandle(process=process, job=job_payload)
        handle.last_message = job_message
        self._processes.append(handle)
        self._last_exit_code = None
        self._last_finished_at = None
        self._last_started_at = handle.started_at
        self._last_message = job_message
        self._last_run_state = None

        if process.stdout is not None:
            thread = threading.Thread(
                target=self._consume_process_output,
                args=(handle, process.stdout),
                name=f"build-log-consumer-{process.pid}",
                daemon=True,
            )
            handle.log_thread = thread
            thread.start()

        if playlist_name:
            self._remove_waiting_playlist_locked(playlist_name)
        self._sync_waiting_from_pending_locked()
        return True, job_message

    def _launch_jobs_if_capacity_locked(self) -> None:
        limit = self._parallel_limit_locked()
        while self._pending_jobs and len(self._processes) < limit:
            next_job = dict(self._pending_jobs[0])
            success, _ = self._launch_job_locked(next_job)
            if success:
                self._pending_jobs.pop(0)
            else:
                # Prevent tight loop if job cannot start.
                self._pending_jobs.pop(0)
                break
        self._sync_waiting_from_pending_locked()


    def _record_job_result_locked(
        self,
        job: Optional[Dict[str, Any]],
        exit_code: Optional[int],
        started_at: Optional[datetime],
        finished_at: Optional[datetime],
        state: str,
        message: Optional[str],
    ) -> None:
        if not job:
            return

        result = {
            "state": state,
            "exit_code": exit_code,
            "message": message,
            "started_at": _format_timestamp(started_at),
            "finished_at": _format_timestamp(finished_at),
        }

        if job.get("type") == "all":
            self._last_all_result = result
        elif job.get("type") == "playlist":
            playlist_name = job.get("playlist")
            if playlist_name:
                self._last_playlist_results[playlist_name] = result

    def _normalize_process_state_locked(self) -> None:
        if not self._processes:
            return

        active_handles: List[_ProcessHandle] = []
        any_finished = False

        for handle in list(self._processes):
            return_code = handle.process.poll()
            if return_code is None:
                active_handles.append(handle)
                continue

            any_finished = True
            finished_at = _utcnow()
            handle.exit_code = return_code
            handle.finished_at = finished_at
            self._last_exit_code = return_code
            self._last_finished_at = finished_at

            job_snapshot = handle.job.copy() if isinstance(handle.job, dict) else {}
            started_at = handle.started_at

            if handle.stop_requested:
                if job_snapshot.get("type") == "playlist" and job_snapshot.get("playlist"):
                    message = f"Build for playlist '{job_snapshot['playlist']}' stopped."
                elif job_snapshot.get("type") == "all":
                    message = "Build for all playlists stopped."
                else:
                    message = "Build process stopped."
                state = "stopped"
            else:
                if return_code == 0:
                    message = "Build completed successfully."
                    state = "success"
                else:
                    message = f"Build exited with code {return_code}."
                    state = "error"

            self._last_message = message
            self._record_job_result_locked(
                job_snapshot,
                return_code,
                started_at,
                finished_at,
                state,
                message,
            )
            self._last_run_state = state

            playlist_name = job_snapshot.get("playlist")
            if playlist_name:
                self._remove_waiting_playlist_locked(playlist_name)
                self._observed_active_playlists.discard(playlist_name)
                self._mark_playlist_completed_locked(playlist_name, finished_at)

        self._processes = active_handles

        if not self._processes:
            if not self._pending_jobs:
                self._queued_playlists = []
            self._observed_active_playlists.clear()
            self._passive_running = False
            self._passive_job = None
            self._passive_last_state = self._last_run_state
            self._passive_last_started_at = None
            self._passive_last_finished_at = self._last_finished_at
            self._passive_last_message = self._last_message

        if any_finished:
            self._sync_waiting_from_pending_locked()
            self._launch_jobs_if_capacity_locked()

    def _status_snapshot_locked(self) -> Dict[str, Any]:
        self._normalize_process_state_locked()
        self._sync_waiting_from_pending_locked()

        running_process = bool(self._processes)
        running_workers = len(self._processes)
        parallel_limit = self._parallel_limit_locked()
        pending_jobs = len(self._pending_jobs)
        passive_active = self._passive_running or bool(self._observed_active_playlists)
        running = running_process or passive_active

        status_label = "running" if running else "idle"
        if not running:
            if self._last_run_state in {"success", "error", "stopped"}:
                status_label = self._last_run_state
            elif self._passive_last_state in {"success", "error", "stopped"}:
                status_label = self._passive_last_state
            elif self._last_exit_code not in (None, 0):
                status_label = "error"

        observed_active_names: Set[str] = set()
        for name in self._observed_active_playlists:
            normalized = str(name).strip()
            if normalized:
                observed_active_names.add(normalized)

        job: Optional[Dict[str, Any]] = None
        if self._processes:
            if len(self._processes) == 1:
                handle = self._processes[0]
                raw_job = handle.job or {}
                job = raw_job.copy()
                if isinstance(job.get("active_playlists"), list):
                    job["active_playlists"] = list(job["active_playlists"])
                job["started_at"] = _format_timestamp(handle.started_at)
                if handle.stop_requested:
                    job["stop_requested"] = True
            else:
                job = {"type": "parallel_playlists", "jobs": []}
                for handle in self._processes:
                    raw_job = handle.job or {}
                    entry = raw_job.copy()
                    if isinstance(entry.get("active_playlists"), list):
                        entry["active_playlists"] = list(entry["active_playlists"])
                    entry["started_at"] = _format_timestamp(handle.started_at)
                    job["jobs"].append(entry)
        elif passive_active:
            job = self._build_passive_job_snapshot_locked()

        job_active_names: Set[str] = set(observed_active_names)

        def _collect_names_from_job(job_payload: Optional[Dict[str, Any]]) -> None:
            if not isinstance(job_payload, dict):
                return
            existing_active = job_payload.get("active_playlists")
            if isinstance(existing_active, list):
                for item in existing_active:
                    normalized = self._normalize_playlist_key(item)
                    if normalized:
                        job_active_names.add(normalized)
            playlist_value = job_payload.get("playlist")
            normalized_playlist = self._normalize_playlist_key(playlist_value)
            if normalized_playlist:
                job_active_names.add(normalized_playlist)

        if self._processes:
            for handle in self._processes:
                _collect_names_from_job(handle.job)

        if job:
            if job.get("type") == "parallel_playlists":
                for entry in job.get("jobs", []):
                    if isinstance(entry, dict):
                        _collect_names_from_job(entry)
            else:
                if isinstance(job.get("progress"), dict):
                    job["progress"] = job["progress"].copy()
                _collect_names_from_job(job)

            if job_active_names:
                job["active_playlists"] = sorted(job_active_names)
            elif "active_playlists" in job:
                job.pop("active_playlists", None)

        waiting_names: List[str] = []
        if self._queued_playlists:
            seen_waiting: Set[str] = set()
            for name in self._queued_playlists:
                normalized_waiting = self._normalize_playlist_key(name)
                if (
                    not normalized_waiting
                    or normalized_waiting in job_active_names
                    or normalized_waiting in seen_waiting
                ):
                    continue
                waiting_names.append(normalized_waiting)
                seen_waiting.add(normalized_waiting)

        if job is not None:
            if waiting_names:
                job["waiting_playlists"] = list(waiting_names)
            elif "waiting_playlists" in job:
                job.pop("waiting_playlists", None)

        self._prune_playlist_logs_locked(job_active_names, waiting_names)

        playlist_results = {
            name: result.copy()
            for name, result in self._last_playlist_results.items()
        }
        playlist_logs: Dict[str, List[Any]] = {}
        for name, entries in self._playlist_logs.items():
            normalized_entries: List[Any] = []
            for entry in entries:
                if isinstance(entry, dict):
                    normalized_entries.append(entry.copy())
                else:
                    normalized_entries.append(entry)
            playlist_logs[name] = normalized_entries

        if running_process:
            start_times = [handle.started_at for handle in self._processes if handle.started_at]
            since_timestamp = min(start_times) if start_times else None
        else:
            since_timestamp = self._passive_last_started_at

        finished_timestamp = (
            self._last_finished_at if running_process else self._passive_last_finished_at
        )
        message_value = self._last_message or self._passive_last_message

        self._prune_general_logs_locked()
        general_log_entries = self._general_logs[-GENERAL_LOG_DISPLAY_LIMIT:]
        general_logs_payload: List[Dict[str, Any]] = []
        for entry in general_log_entries:
            if isinstance(entry, dict):
                text_value = entry.get("text")
                if not text_value:
                    continue
                payload_entry: Dict[str, Any] = {"text": str(text_value)}
                timestamp_value = entry.get("timestamp")
                if isinstance(timestamp_value, datetime):
                    payload_entry["timestamp"] = _format_timestamp(timestamp_value)
                elif isinstance(timestamp_value, str):
                    payload_entry["timestamp"] = timestamp_value

                general_logs_payload.append(payload_entry)
            elif isinstance(entry, str):
                general_logs_payload.append({"text": entry})

        active_playlists_payload = sorted(job_active_names)

        if running_process and len(self._processes) == 1:
            pid_value: Optional[int] = self._processes[0].process.pid
        else:
            pid_value = None

        return {
            "running": running,
            "status": status_label,
            "pid": pid_value,
            "since": _format_timestamp(since_timestamp),
            "last_finished": _format_timestamp(finished_timestamp),
            "exit_code": self._last_exit_code,
            "message": message_value,
            "job": job,
            "active_playlists": active_playlists_payload,
            "waiting_playlists": waiting_names,
            "parallel_limit": parallel_limit,
            "running_workers": running_workers,
            "pending_jobs": pending_jobs,
            "results": {
                "all": self._last_all_result.copy() if self._last_all_result else None,
                "playlists": playlist_results,
            },
            "logs": {
                "playlists": playlist_logs,
                "general": general_logs_payload,
            },
        }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return self._status_snapshot_locked()

    def start(self, playlist: Optional[str] = None) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        with self._lock:
            self._normalize_process_state_locked()

            playlist_name: Optional[str]
            if playlist is None:
                playlist_name = None
            else:
                playlist_name = str(playlist).strip() or None
                if playlist_name is None:
                    message = "Playlist name is required to start a build."
                    self._last_message = message
                    return False, self._status_snapshot_locked(), message

            if playlist_name:
                job: Dict[str, Any] = {"type": "playlist", "playlist": playlist_name}
            else:
                job = {"type": "all"}

            parallel_limit = self._parallel_limit_locked()
            if len(self._processes) >= parallel_limit:
                self._pending_jobs.append(dict(job))
                if job.get("type") == "playlist":
                    queue_message = f"Build for playlist '{playlist_name}' queued."
                    self._append_playlist_log_locked(
                        playlist_name, "Queued – waiting to start…"
                    )
                else:
                    queue_message = "Build for all playlists queued."
                self._last_message = queue_message
                self._sync_waiting_from_pending_locked()
                status = self._status_snapshot_locked()
                return True, status, queue_message

            if not self._processes:
                self._observed_active_playlists.clear()
                self._passive_running = False
                self._passive_job = None
                self._passive_last_started_at = None
                self._passive_last_finished_at = None
                self._passive_last_message = None
                self._passive_last_state = None
                self._queued_playlists = []

            success, message = self._launch_job_locked(job)
            if not success:
                status = self._status_snapshot_locked()
                status["status"] = "error"
                return False, status, message

            self._launch_jobs_if_capacity_locked()
            status = self._status_snapshot_locked()
            return True, status, message

    def stop(self, timeout: float = 10.0) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        with self._lock:
            self._normalize_process_state_locked()
            if not self._processes:
                message = "Builder is not running."
                self._last_message = message
                return False, self._status_snapshot_locked(), message

            handles = list(self._processes)
            if len(handles) == 1:
                job = handles[0].job if isinstance(handles[0].job, dict) else {}
                if job.get("type") == "playlist" and job.get("playlist"):
                    stopping_message = f"Stopping build for playlist '{job['playlist']}'."
                elif job.get("type") == "all":
                    stopping_message = "Stopping build for all playlists."
                else:
                    stopping_message = "Stopping build."
            else:
                stopping_message = "Stopping all active builds."

            self._last_message = stopping_message
            for handle in handles:
                handle.stop_requested = True
            self._pending_jobs.clear()
            self._queued_playlists = []
            self._sync_waiting_from_pending_locked()

        encountered_error: Optional[Exception] = None
        for handle in handles:
            try:
                handle.process.terminate()
                try:
                    handle.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    handle.process.kill()
                    handle.process.wait(timeout=5)
            except Exception as exc:  # pragma: no cover - defensive
                encountered_error = exc
                break

        if encountered_error is not None:
            with self._lock:
                error_message = f"Unable to stop build: {encountered_error}"
                self._last_message = error_message
                status = self._status_snapshot_locked()
                status["status"] = "error"
                return False, status, error_message

        with self._lock:
            self._normalize_process_state_locked()
            message = self._last_message or "Build process stopped."
            status = self._status_snapshot_locked()
            return True, status, message

    def stop_playlist(
        self, playlist: Optional[str], timeout: float = 10.0
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        normalized = self._normalize_playlist_key(playlist)
        if not normalized:
            with self._lock:
                message = "Playlist name is required to stop a build."
                self._last_message = message
                status = self._status_snapshot_locked()
                return False, status, message

        handles_to_stop: List[_ProcessHandle] = []
        removed_from_pending = False
        removed_from_queue = False

        with self._lock:
            self._normalize_process_state_locked()

            remaining_jobs: List[Dict[str, Any]] = []
            for job in self._pending_jobs:
                if job.get("type") == "playlist":
                    candidate = self._normalize_playlist_key(job.get("playlist"))
                    if candidate == normalized:
                        removed_from_pending = True
                        continue
                remaining_jobs.append(job)
            self._pending_jobs = remaining_jobs

            queued_snapshot = list(self._queued_playlists)
            self._remove_waiting_playlist_locked(normalized)
            removed_from_queue = queued_snapshot != self._queued_playlists

            handle = self._find_handle_for_playlist_locked(normalized)
            if handle is None:
                all_handle = self._find_handle_for_all_locked()
                normalized_active = {
                    self._normalize_playlist_key(name)
                    for name in self._observed_active_playlists
                }
                if (
                    all_handle is not None
                    and normalized in normalized_active
                ):
                    if all_handle not in handles_to_stop:
                        handles_to_stop.append(all_handle)
                    all_handle.stop_requested = True
                    stopping_message = (
                        f"Stopping build for playlist '{normalized}' as part of the all-playlists job."
                    )
                    self._last_message = stopping_message
                    self._pending_jobs.clear()
                    self._queued_playlists = []
                    self._sync_waiting_from_pending_locked()
                else:
                    self._sync_waiting_from_pending_locked()
                    if removed_from_pending or removed_from_queue:
                        message = f"Build for playlist '{normalized}' cancelled."
                        self._last_message = message
                        status = self._status_snapshot_locked()
                        return True, status, message

                    message = f"No active build found for playlist '{normalized}'."
                    self._last_message = message
                    status = self._status_snapshot_locked()
                    return False, status, message
            else:
                handle.stop_requested = True
                if handle not in handles_to_stop:
                    handles_to_stop.append(handle)
                stopping_message = f"Stopping build for playlist '{normalized}'."
                self._last_message = stopping_message
                self._sync_waiting_from_pending_locked()

        encountered_error: Optional[Exception] = None
        for handle in handles_to_stop:
            try:
                handle.process.terminate()
                try:
                    handle.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    handle.process.kill()
                    handle.process.wait(timeout=5)
            except Exception as exc:  # pragma: no cover - defensive
                encountered_error = exc
                break

        with self._lock:
            if encountered_error is not None:
                error_message = (
                    f"Unable to stop build for playlist '{normalized}': {encountered_error}"
                )
                self._last_message = error_message
                status = self._status_snapshot_locked()
                status["status"] = "error"
                return False, status, error_message

            self._normalize_process_state_locked()
            completion_message = f"Build for playlist '{normalized}' cancelled."
            self._last_message = completion_message
            status = self._status_snapshot_locked()
            return True, status, completion_message


def humanize_field_name(field: str) -> str:
    if field in FIELD_LABEL_OVERRIDES:
        return FIELD_LABEL_OVERRIDES[field]

    label = field.replace(".", " ").replace("_", " ")
    label = re.sub(r"(?<!^)(?=[A-Z])", " ", label)
    words = [w.upper() if w.lower() in {"id", "guid", "uuid"} else w for w in label.split()]
    humanized = " ".join(word.capitalize() if word.islower() else word for word in words)
    return humanized


def load_field_options() -> List[Dict[str, str]]:
    return [
        {"value": field, "label": humanize_field_name(field)}
        for field in CURATED_FIELD_CHOICES
    ]


def resolve_allmusic_cache_path() -> Path:
    """Return the filesystem path used for the AllMusic popularity cache."""

    cache_path = DEFAULT_ALLMUSIC_CACHE

    config_data = _load_config_data()
    allmusic_cfg = config_data.get("allmusic") or {}
    if isinstance(allmusic_cfg, dict):
        cache_path = _resolve_path_setting(allmusic_cfg.get("cache_file"), cache_path)

    return cache_path


def resolve_log_file_path() -> Optional[Path]:
    """Resolve the configured log file path for the playlist builder."""

    log_path: Optional[Path] = DEFAULT_LOG_PATH

    config_data = _load_config_data()
    logging_cfg = config_data.get("logging") or {}
    if isinstance(logging_cfg, dict):
        log_path = _resolve_path_setting(logging_cfg.get("file"), log_path)

    return log_path


def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def load_yaml_data() -> Dict[str, Any]:
    if not PLAYLISTS_PATH.exists():
        return {"defaults": {"plex_filter": []}, "playlists": OrderedDict()}

    with PLAYLISTS_PATH.open("r", encoding="utf-8") as playlist_file:
        data = yaml.safe_load(playlist_file) or {}

    defaults = data.get("defaults", {}) or {}
    raw_playlists = data.get("playlists", {}) or {}
    playlist_items: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(raw_playlists, dict):
        for name, config in raw_playlists.items():
            if not isinstance(name, str):
                continue
            playlist_items.append((name, config or {}))
    playlists = _alphabetize_playlist_configs(playlist_items)
    return {"defaults": defaults, "playlists": playlists}


def _normalize_bool_flag(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False

    if value is None:
        return default

    return bool(value)


def normalize_filter_entry(filter_entry: Dict[str, Any]) -> Dict[str, Any]:
    field = filter_entry.get("field", "")
    operator = filter_entry.get("operator", "equals")
    value = filter_entry.get("value", "")
    match_all = filter_entry.get("match_all")
    wildcard = filter_entry.get("wildcard")

    if isinstance(value, list):
        value_str = ", ".join(str(item) for item in value)
    else:
        value_str = "" if value is None else str(value)

    return {
        "field": field,
        "operator": operator,
        "value": value_str,
        "match_all": _normalize_bool_flag(match_all, default=True),
        "wildcard": _normalize_bool_flag(wildcard, default=False),
    }


def serialize_filters(filters: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not filters:
        return []
    return [normalize_filter_entry(filter_entry or {}) for filter_entry in filters]


def normalize_boost_condition_entry(condition_entry: Dict[str, Any]) -> Dict[str, Any]:
    field = condition_entry.get("field", "")
    operator = condition_entry.get("operator", "equals")
    value = condition_entry.get("value", "")
    match_all_raw = condition_entry.get("match_all")

    has_multiple_expected_values = False

    if isinstance(value, list):
        value_str = ", ".join(str(item) for item in value)
        has_multiple_expected_values = len(value) > 1
    else:
        value_str = "" if value is None else str(value)
        if isinstance(value, str) and "," in value:
            segments = [segment.strip() for segment in value.split(",")]
            has_multiple_expected_values = len([segment for segment in segments if segment]) > 1

    match_all_value = _normalize_bool_flag(match_all_raw, default=True)
    if has_multiple_expected_values:
        match_all_value = False

    return {
        "field": field,
        "operator": operator,
        "value": value_str,
        "match_all": match_all_value,
    }


def normalize_boost_entry(boost_entry: Dict[str, Any]) -> Dict[str, Any]:
    boost_raw = boost_entry.get("boost", 1.0)

    try:
        boost_value = float(boost_raw)
    except (TypeError, ValueError):
        boost_value = 1.0

    if not math.isfinite(boost_value) or boost_value < 0:
        boost_value = 1.0

    raw_conditions = boost_entry.get("conditions")
    if isinstance(raw_conditions, list):
        normalized_conditions = [
            normalize_boost_condition_entry(condition or {})
            for condition in raw_conditions
        ]
    else:
        normalized_conditions = [normalize_boost_condition_entry(boost_entry)]

    if not normalized_conditions:
        normalized_conditions = [normalize_boost_condition_entry({})]

    return {
        "conditions": normalized_conditions,
        "boost": boost_value,
    }


def serialize_boosts(boosts: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not boosts:
        return []
    return [normalize_boost_entry(boost_entry or {}) for boost_entry in boosts]


def load_playlists() -> Dict[str, Any]:
    data = load_yaml_data()
    defaults_config = data.get("defaults", {}) or {}
    defaults_filters = serialize_filters(defaults_config.get("plex_filter"))
    defaults_boosts = serialize_boosts(defaults_config.get("popularity_boosts"))
    defaults_extras = {
        key: value
        for key, value in defaults_config.items()
        if key not in {"plex_filter", "popularity_boosts"}
    }

    playlists_data = []
    for name, config in data.get("playlists", {}).items():
        config = config or {}
        extras = {
            key: value
            for key, value in config.items()
            if key
            not in {
                "limit",
                "artist_limit",
                "album_limit",
                "year_limit",
                "sort_by",
                "after_sort",
                "plex_filter",
                "top_5_boost",
                "popularity_boosts",
                "source",
                "spotify_url",
            }
        }
        raw_source = config.get("source")
        if isinstance(raw_source, str):
            normalized_source = raw_source.strip().lower()
        else:
            normalized_source = ""
        if normalized_source not in {"plex", "spotify"}:
            normalized_source = "plex"

        raw_spotify_url = config.get("spotify_url")
        if raw_spotify_url is None:
            spotify_url = ""
        else:
            spotify_url = str(raw_spotify_url).strip()
        if normalized_source != "spotify":
            spotify_url = ""

        playlists_data.append(
            {
                "name": name,
                "limit": config.get("limit", 0) or 0,
                "artist_limit": config.get("artist_limit", 0) or 0,
                "album_limit": config.get("album_limit", 0) or 0,
                "year_limit": config.get("year_limit", 0) or 0,
                "sort_by": config.get("sort_by", ""),
                "after_sort": config.get("after_sort", "") or "",
                "plex_filter": serialize_filters(config.get("plex_filter")),
                "popularity_boosts": serialize_boosts(
                    config.get("popularity_boosts")
                ),
                "extras": extras,
                "source": normalized_source,
                "spotify_url": spotify_url,
            }
        )

    return {
        "defaults": {
            "plex_filter": defaults_filters,
            "popularity_boosts": defaults_boosts,
            "extras": defaults_extras,
        },
        "playlists": playlists_data,
    }


def parse_filter_value(raw_value: str) -> Any:
    if raw_value is None:
        return ""
    if isinstance(raw_value, list):
        return raw_value

    value = str(raw_value).strip()
    if not value:
        return ""

    # Treat comma-separated values as lists
    if "," in value:
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items

    # Try numeric conversions
    if value.isdigit():
        return int(value)
    try:
        float_value = float(value)
    except ValueError:
        float_value = None
    if float_value is not None and "." in value:
        return float_value

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    return value


def build_filter_for_yaml(filter_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    field = (filter_entry.get("field") or "").strip()
    if not field:
        return None
    operator = filter_entry.get("operator", "equals").strip() or "equals"
    value = parse_filter_value(filter_entry.get("value", ""))
    match_all = filter_entry.get("match_all", True)
    wildcard = filter_entry.get("wildcard", False)

    yaml_entry: Dict[str, Any] = {
        "field": field,
        "operator": operator,
        "value": value,
    }

    if isinstance(value, list):
        if isinstance(match_all, bool) and not match_all:
            yaml_entry["match_all"] = False
    elif isinstance(match_all, bool) and not match_all:
        yaml_entry["match_all"] = False

    if isinstance(wildcard, bool) and wildcard:
        yaml_entry["wildcard"] = True

    return yaml_entry


def build_boost_condition_for_yaml(
    condition_entry: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    field = (condition_entry.get("field") or "").strip()
    if not field:
        return None

    operator = (condition_entry.get("operator") or "equals").strip() or "equals"
    value = parse_filter_value(condition_entry.get("value", ""))
    match_all_flag = _normalize_bool_flag(condition_entry.get("match_all"), default=True)

    yaml_condition: Dict[str, Any] = {
        "field": field,
        "operator": operator,
        "value": value,
    }

    if isinstance(value, list):
        if isinstance(match_all_flag, bool) and not match_all_flag:
            yaml_condition["match_all"] = False
    elif isinstance(match_all_flag, bool) and not match_all_flag:
        yaml_condition["match_all"] = False

    return yaml_condition


def build_boost_for_yaml(boost_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    boost_raw = boost_entry.get("boost", 1.0)
    boost_value = to_float(boost_raw, default=1.0)
    if boost_value < 0:
        boost_value = 1.0

    raw_conditions = boost_entry.get("conditions")
    if isinstance(raw_conditions, list):
        condition_candidates = raw_conditions
    else:
        condition_candidates = [boost_entry]

    yaml_conditions = []
    for condition_entry in condition_candidates:
        yaml_condition = build_boost_condition_for_yaml(condition_entry or {})
        if yaml_condition is not None:
            yaml_conditions.append(yaml_condition)

    if not yaml_conditions:
        return None

    return {
        "conditions": yaml_conditions,
        "boost": boost_value,
    }


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 1.0) -> float:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return default

    if not math.isfinite(numeric):
        return default

    return numeric


def save_playlists(payload: Dict[str, Any]) -> None:
    defaults_payload = payload.get("defaults", {}) or {}
    playlists_payload = payload.get("playlists", []) or []

    defaults_filters = []
    for filter_entry in defaults_payload.get("plex_filter", []):
        yaml_filter = build_filter_for_yaml(filter_entry)
        if yaml_filter is not None:
            defaults_filters.append(yaml_filter)

    defaults_boosts = []
    for boost_entry in defaults_payload.get("popularity_boosts", []):
        yaml_boost = build_boost_for_yaml(boost_entry)
        if yaml_boost is not None:
            defaults_boosts.append(yaml_boost)

    defaults_config: Dict[str, Any] = {}
    extras = defaults_payload.get("extras")
    if isinstance(extras, dict):
        defaults_config.update(
            {
                key: value
                for key, value in extras.items()
                if key not in {"source", "spotify_url"}
            }
        )
    if defaults_filters:
        defaults_config["plex_filter"] = defaults_filters
    if defaults_boosts:
        defaults_config["popularity_boosts"] = defaults_boosts

    playlists_dict: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for playlist_entry in playlists_payload:
        name = (playlist_entry.get("name") or "").strip()
        if not name:
            continue

        limit = to_int(playlist_entry.get("limit", 0))
        artist_limit = to_int(playlist_entry.get("artist_limit", 0))
        album_limit = to_int(playlist_entry.get("album_limit", 0))
        year_limit = to_int(playlist_entry.get("year_limit", 0))
        sort_by = playlist_entry.get("sort_by") or None
        after_sort = playlist_entry.get("after_sort") or None
        playlist_config: Dict[str, Any] = {}
        extras = playlist_entry.get("extras")
        if isinstance(extras, dict):
            playlist_config.update(
                {
                    key: value
                    for key, value in extras.items()
                    if key not in {"source", "spotify_url"}
                }
            )
        playlist_config["limit"] = max(limit, 0)
        playlist_config["artist_limit"] = max(artist_limit, 0)
        playlist_config["album_limit"] = max(album_limit, 0)
        playlist_config["year_limit"] = max(year_limit, 0)
        if sort_by:
            playlist_config["sort_by"] = sort_by
        if after_sort:
            playlist_config["after_sort"] = after_sort

        raw_source = playlist_entry.get("source")
        if isinstance(raw_source, str):
            normalized_source = raw_source.strip().lower()
        else:
            normalized_source = "plex"
        if normalized_source not in {"plex", "spotify"}:
            normalized_source = "plex"
        if normalized_source != "plex":
            playlist_config["source"] = normalized_source
        else:
            playlist_config.pop("source", None)

        raw_spotify_url = playlist_entry.get("spotify_url")
        spotify_url = "" if raw_spotify_url is None else str(raw_spotify_url).strip()
        if normalized_source == "spotify" and spotify_url:
            playlist_config["spotify_url"] = spotify_url
        else:
            playlist_config.pop("spotify_url", None)

        playlist_filters = []
        for filter_entry in playlist_entry.get("plex_filter", []):
            yaml_filter = build_filter_for_yaml(filter_entry)
            if yaml_filter is not None:
                playlist_filters.append(yaml_filter)
        if playlist_filters:
            playlist_config["plex_filter"] = playlist_filters

        boost_entries = []
        for boost_entry in playlist_entry.get("popularity_boosts", []):
            yaml_boost = build_boost_for_yaml(boost_entry)
            if yaml_boost is not None:
                boost_entries.append(yaml_boost)
        if boost_entries:
            playlist_config["popularity_boosts"] = boost_entries

        playlists_dict[name] = playlist_config

    yaml_structure: Dict[str, Any] = {}
    yaml_structure["defaults"] = defaults_config
    yaml_structure["playlists"] = _alphabetize_playlist_configs(
        playlists_dict.items()
    )

    with PLAYLISTS_PATH.open("w", encoding="utf-8") as playlist_file:
        yaml.safe_dump(yaml_structure, playlist_file, sort_keys=False, allow_unicode=True)


def save_single_playlist(
    playlist_payload: Dict[str, Any], original_name: Optional[str] = None
) -> str:
    playlist_name = (playlist_payload.get("name") or "").strip()
    if not playlist_name:
        raise ValueError("Playlist name is required.")

    extras = playlist_payload.get("extras")
    playlist_config: Dict[str, Any] = {}
    if isinstance(extras, dict):
        playlist_config.update(
            {
                key: value
                for key, value in extras.items()
                if key not in {"source", "spotify_url"}
            }
        )

    limit = to_int(playlist_payload.get("limit", 0))
    artist_limit = to_int(playlist_payload.get("artist_limit", 0))
    album_limit = to_int(playlist_payload.get("album_limit", 0))
    year_limit = to_int(playlist_payload.get("year_limit", 0))
    sort_by = (playlist_payload.get("sort_by") or "").strip() or None
    after_sort = (playlist_payload.get("after_sort") or "").strip() or None
    playlist_config["limit"] = max(limit, 0)
    playlist_config["artist_limit"] = max(artist_limit, 0)
    playlist_config["album_limit"] = max(album_limit, 0)
    playlist_config["year_limit"] = max(year_limit, 0)
    if sort_by:
        playlist_config["sort_by"] = sort_by
    if after_sort:
        playlist_config["after_sort"] = after_sort

    raw_source = playlist_payload.get("source")
    if isinstance(raw_source, str):
        normalized_source = raw_source.strip().lower()
    else:
        normalized_source = "plex"
    if normalized_source not in {"plex", "spotify"}:
        normalized_source = "plex"
    if normalized_source != "plex":
        playlist_config["source"] = normalized_source
    else:
        playlist_config.pop("source", None)

    raw_spotify_url = playlist_payload.get("spotify_url")
    spotify_url = "" if raw_spotify_url is None else str(raw_spotify_url).strip()
    if normalized_source == "spotify" and spotify_url:
        playlist_config["spotify_url"] = spotify_url
    else:
        playlist_config.pop("spotify_url", None)

    playlist_filters = []
    for filter_entry in playlist_payload.get("plex_filter", []):
        yaml_filter = build_filter_for_yaml(filter_entry)
        if yaml_filter is not None:
            playlist_filters.append(yaml_filter)
    if playlist_filters:
        playlist_config["plex_filter"] = playlist_filters

    boost_entries = []
    for boost_entry in playlist_payload.get("popularity_boosts", []):
        yaml_boost = build_boost_for_yaml(boost_entry)
        if yaml_boost is not None:
            boost_entries.append(yaml_boost)
    if boost_entries:
        playlist_config["popularity_boosts"] = boost_entries

    yaml_data = load_yaml_data()
    defaults_config = yaml_data.get("defaults", {}) or {}
    existing_playlists = yaml_data.get("playlists", OrderedDict()) or OrderedDict()
    if not isinstance(existing_playlists, OrderedDict):
        existing_playlists = OrderedDict(existing_playlists.items())

    normalized_original = (original_name or "").strip()
    updated_playlists: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    inserted = False
    conflict = False

    for existing_name, existing_config in existing_playlists.items():
        if normalized_original:
            if existing_name == normalized_original:
                updated_playlists[playlist_name] = playlist_config
                inserted = True
                continue
        else:
            if existing_name == playlist_name:
                updated_playlists[playlist_name] = playlist_config
                inserted = True
                continue

        if existing_name == playlist_name:
            conflict = True

        updated_playlists[existing_name] = existing_config

    if conflict:
        raise PlaylistConflictError(
            f"Playlist '{playlist_name}' already exists. Please choose a different name."
        )

    if not inserted:
        updated_playlists[playlist_name] = playlist_config

    yaml_structure: Dict[str, Any] = {
        "defaults": defaults_config,
        "playlists": _alphabetize_playlist_configs(updated_playlists.items()),
    }

    with PLAYLISTS_PATH.open("w", encoding="utf-8") as playlist_file:
        yaml.safe_dump(yaml_structure, playlist_file, sort_keys=False, allow_unicode=True)

    return playlist_name


def _determine_separator(raw_path: str) -> str:
    if raw_path.count("\\") > raw_path.count("/"):
        return "\\"
    return "/"


IMAGE_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".avif",
}


def _is_image_file(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
    except OSError:
        return False

    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type and mime_type.startswith("image/"):
        return True

    return path.suffix.lower() in IMAGE_EXTENSIONS


def _cover_upload_directories() -> List[Path]:
    candidates: List[Path] = []
    env_override = os.environ.get("PMB_COVER_UPLOAD_DIR")
    if env_override:
        candidate = Path(env_override).expanduser()
        if not candidate.is_absolute():
            candidate = (CONFIG_DIR / candidate).resolve()
        candidates.append(candidate)

    candidates.append(Path("/images"))
    candidates.append((CONFIG_DIR / "images").resolve())

    return candidates


def _ensure_cover_upload_dir() -> Path:
    last_error: Optional[Exception] = None
    for candidate in _cover_upload_directories():
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            last_error = exc
            continue

        try:
            return candidate.resolve()
        except OSError:
            return candidate

    raise OSError("Unable to prepare cover upload directory.") from last_error


def _generate_unique_cover_name(directory: Path, base_name: str, suffix: str) -> str:
    sanitized_base = base_name or "cover"
    sanitized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    candidate = f"{sanitized_base}{sanitized_suffix}"
    if not (directory / candidate).exists():
        return candidate

    for index in range(1, 1000):
        candidate = f"{sanitized_base}_{index}{sanitized_suffix}"
        if not (directory / candidate).exists():
            return candidate

    unique_fragment = uuid.uuid4().hex[:8]
    return f"{sanitized_base}_{unique_fragment}{sanitized_suffix}"


def _format_cover_upload_path(path: Path) -> str:
    try:
        resolved_path = path.resolve()
    except OSError:
        resolved_path = path

    try:
        base_dir = PLAYLISTS_PATH.parent.resolve()
    except OSError:
        base_dir = PLAYLISTS_PATH.parent

    try:
        relative = resolved_path.relative_to(base_dir)
    except ValueError:
        return resolved_path.as_posix()

    return relative.as_posix()


def _find_existing_directory(path: Path) -> Optional[Path]:
    current = path
    visited = set()
    while True:
        if current in visited:
            break
        visited.add(current)
        if current.exists() and current.is_dir():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def resolve_directory_request(raw_path: str) -> Tuple[Path, str, str]:
    trimmed = (raw_path or "").strip()
    separator = _determine_separator(trimmed) if trimmed else os.sep

    base_dir = PLAYLISTS_PATH.parent
    if not trimmed:
        return base_dir, "", separator

    expanded = os.path.expanduser(trimmed)
    path = Path(expanded)
    if not path.is_absolute():
        path = (PLAYLISTS_PATH.parent / path).resolve()
    else:
        path = path.resolve()

    if trimmed.endswith(("/", "\\")):
        directory_candidate = path
        prefix = trimmed
    else:
        directory_candidate = path.parent
        last_slash = max(trimmed.rfind("/"), trimmed.rfind("\\"))
        prefix = trimmed[: last_slash + 1] if last_slash >= 0 else ""

    existing_directory = _find_existing_directory(directory_candidate)
    if existing_directory is None:
        return base_dir, "", separator

    if prefix and not prefix.endswith(("/", "\\")):
        prefix = f"{prefix}{separator}"

    return existing_directory, prefix, separator


def resolve_file_request(raw_path: str) -> Optional[Path]:
    trimmed = (raw_path or "").strip()
    if not trimmed:
        return None

    expanded = os.path.expanduser(trimmed)
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = (PLAYLISTS_PATH.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()

    try:
        if candidate.exists() and candidate.is_file():
            return candidate
    except OSError:
        return None

    return None


def create_app() -> Flask:
    app = Flask(__name__)

    field_options = load_field_options()
    operator_options = [
        {"value": value, "label": label} for value, label in OPERATOR_OPTIONS.items()
    ]
    sort_options = [
        {"value": value, "label": label} for value, label in SORT_OPTIONS.items()
    ]

    build_manager = BuildManager([sys.executable, "main.py"], BASE_DIR)

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/api/build/status", methods=["GET"])
    def build_status() -> Any:
        return jsonify(build_manager.get_status())

    @app.route("/api/build/start", methods=["POST"])
    def start_build() -> Any:
        payload = request.get_json(silent=True)
        playlist_name: Optional[str] = None
        if isinstance(payload, dict) and "playlist" in payload:
            raw_value = payload.get("playlist")
            if raw_value is not None:
                playlist_candidate = str(raw_value).strip()
                if not playlist_candidate:
                    current_status = build_manager.get_status()
                    return (
                        jsonify(
                            {
                                "status": current_status,
                                "message": "Playlist name is required to start a build.",
                            }
                        ),
                        400,
                    )
                playlist_name = playlist_candidate

        started, status, message = build_manager.start(playlist=playlist_name)
        response: Dict[str, Any] = {"status": status}
        if message:
            response["message"] = message
        http_status = 200
        if not started:
            if message == "Builder is already running.":
                http_status = 409
            elif message == "Playlist name is required to start a build.":
                http_status = 400
            else:
                http_status = 500
        return jsonify(response), http_status

    @app.route("/api/build/stop", methods=["POST"])
    def stop_build() -> Any:
        stopped, status, message = build_manager.stop()
        response: Dict[str, Any] = {"status": status}
        if message:
            response["message"] = message
        if stopped:
            http_status = 200
        else:
            http_status = 409 if message == "Builder is not running." else 500
        return jsonify(response), http_status

    @app.route("/api/build/stop_playlist", methods=["POST"])
    def stop_playlist_build() -> Any:
        payload = request.get_json(force=True, silent=True)
        if not isinstance(payload, dict):
            return (
                jsonify({"message": "Playlist name is required to cancel a build."}),
                400,
            )

        raw_value = payload.get("playlist")
        playlist_name = str(raw_value).strip() if raw_value is not None else ""
        if not playlist_name:
            status = build_manager.get_status()
            return (
                jsonify(
                    {
                        "status": status,
                        "message": "Playlist name is required to cancel a build.",
                    }
                ),
                400,
            )

        stopped, status, message = build_manager.stop_playlist(playlist_name)
        response: Dict[str, Any] = {"status": status}
        if message:
            response["message"] = message
        if stopped:
            http_status = 200
        else:
            if message == "Playlist name is required to stop a build.":
                http_status = 400
            elif message and message.startswith("No active build found for playlist"):
                http_status = 404
            else:
                http_status = 500
        return jsonify(response), http_status

    @app.route("/api/config", methods=["GET"])
    def get_config_settings() -> Any:
        payload = _build_config_sections_payload()
        payload["supports_restart"] = True
        return jsonify(payload)

    @app.route("/api/config", methods=["POST"])
    def update_config_settings() -> Any:
        payload = request.get_json(force=True, silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON payload."}), 400

        sections_payload = payload.get("sections")
        if not isinstance(sections_payload, dict):
            return jsonify({"error": "Invalid configuration payload."}), 400

        config_data = _load_config_data()
        updated_config = dict(config_data)
        changes_applied = False

        try:
            for section_id, definition in CONFIG_FORM_DEFINITION.items():
                raw_section = sections_payload.get(section_id)
                if raw_section is None:
                    continue
                if not isinstance(raw_section, dict):
                    raise ValueError(f"Invalid payload for section '{section_id}'.")

                field_updates: Dict[str, Any] = {}
                for field in definition.get("fields", []):
                    key = field.get("key")
                    if not key or key not in raw_section:
                        continue
                    coerced_value = _coerce_config_field_value(raw_section[key], field)
                    field_updates[key] = coerced_value

                if field_updates:
                    updated_config = _apply_section_updates(
                        updated_config, section_id, field_updates
                    )
                    changes_applied = True
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - defensive guard
            return jsonify({"error": str(exc)}), 400

        if changes_applied and updated_config != config_data:
            saved_config = _save_config_data(updated_config)
        else:
            saved_config = config_data

        response_payload = _build_config_sections_payload(saved_config)
        response_payload["supports_restart"] = True
        response_payload["message"] = (
            "Configuration updated successfully."
            if changes_applied
            else "No configuration changes detected."
        )
        return jsonify(response_payload)

    @app.route("/api/runtime/restart", methods=["POST"])
    def restart_runtime() -> Any:
        try:
            build_manager.stop(timeout=5.0)
        except Exception:
            pass

        _schedule_restart(1.0)
        return (
            jsonify(
                {
                    "message": "Restart requested. The application will exit shortly.",
                    "supports_restart": True,
                }
            ),
            202,
        )

    @app.route("/api/playlists", methods=["GET"])
    def get_playlists() -> Any:
        playlists_data = load_playlists()
        response = {
            "defaults": playlists_data["defaults"],
            "playlists": playlists_data["playlists"],
            "options": {
                "fields": field_options,
                "operators": operator_options,
                "sort_fields": sort_options,
            },
        }
        return jsonify(response)

    @app.route("/api/list_directory", methods=["GET"])
    def list_directory() -> Any:
        raw_path = request.args.get("path", "") or ""
        directory, prefix, separator = resolve_directory_request(raw_path)

        entries: List[Dict[str, Any]] = []
        try:
            candidates = sorted(
                directory.iterdir(),
                key=lambda candidate: (not candidate.is_dir(), candidate.name.lower()),
            )
        except OSError:
            candidates = []

        for candidate in candidates:
            try:
                is_dir = candidate.is_dir()
            except OSError:
                continue

            display_name = f"{candidate.name}{separator}" if is_dir else candidate.name
            if prefix and not prefix.endswith(("/", "\\")):
                suggestion_prefix = f"{prefix}{separator}"
            else:
                suggestion_prefix = prefix
            suggestion = f"{suggestion_prefix}{candidate.name}"
            if is_dir:
                suggestion = f"{suggestion}{separator}"

            entries.append(
                {
                    "name": candidate.name,
                    "is_dir": is_dir,
                    "display": display_name,
                    "suggestion": suggestion,
                    "is_image": False if is_dir else _is_image_file(candidate),
                }
            )

        return jsonify({
            "directory": str(directory),
            "entries": entries,
        })

    @app.route("/api/upload_cover", methods=["POST"])
    def upload_cover() -> Any:
        file_storage = request.files.get("file")
        if file_storage is None or not file_storage.filename:
            return jsonify({"error": "No file provided."}), 400

        filename = secure_filename(file_storage.filename or "")
        if not filename:
            return jsonify({"error": "Invalid file name."}), 400

        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            return (
                jsonify(
                    {
                        "error": "Unsupported image format. Please upload a valid image file.",
                    }
                ),
                400,
            )

        try:
            upload_dir = _ensure_cover_upload_dir()
        except OSError:
            return (
                jsonify({"error": "Unable to prepare cover upload directory."}),
                500,
            )

        base_name = Path(filename).stem or "cover"
        unique_name = _generate_unique_cover_name(upload_dir, base_name, suffix)
        target_path = upload_dir / unique_name

        try:
            file_storage.save(target_path)
        except Exception as exc:  # pragma: no cover - protective logging only
            app.logger.exception("Failed to save uploaded cover image: %s", exc)
            return (
                jsonify({"error": "Failed to save uploaded cover image."}),
                500,
            )

        if not _is_image_file(target_path):
            try:
                target_path.unlink()
            except OSError:
                pass
            return (
                jsonify({"error": "Unsupported image format."}),
                400,
            )

        response_path = _format_cover_upload_path(target_path)

        return (
            jsonify(
                {
                    "status": "uploaded",
                    "path": response_path,
                    "filename": unique_name,
                }
            ),
            201,
        )

    @app.route("/api/cover_preview", methods=["GET"])
    def cover_preview() -> Any:
        raw_path = request.args.get("path", "") or ""
        resolved_path = resolve_file_request(raw_path)
        if resolved_path is None:
            abort(404)

        if not _is_image_file(resolved_path):
            abort(404)

        mime_type, _ = mimetypes.guess_type(str(resolved_path))

        try:
            return send_file(
                resolved_path,
                mimetype=mime_type or "application/octet-stream",
                conditional=True,
            )
        except FileNotFoundError:
            abort(404)
        except PermissionError:
            abort(403)

    @app.route("/api/playlists/save_single", methods=["POST"])
    def save_single_playlist_route() -> Any:
        payload = request.get_json(force=True, silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON payload."}), 400

        playlist_payload = payload.get("playlist")
        if not isinstance(playlist_payload, dict):
            return jsonify({"error": "Playlist data is required."}), 400

        original_name = payload.get("original_name")

        try:
            saved_name = save_single_playlist(playlist_payload, original_name)
        except PlaylistConflictError as exc:
            return jsonify({"error": str(exc)}), 409
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - protective logging only
            return jsonify({"error": str(exc)}), 500

        return jsonify({"status": "saved", "name": saved_name})

    @app.route("/api/playlists", methods=["POST"])
    def write_playlists() -> Any:
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            return jsonify({"error": "Invalid JSON payload."}), 400

        try:
            save_playlists(payload)
        except Exception as exc:  # pragma: no cover - protective logging only
            return jsonify({"error": str(exc)}), 500

        return jsonify({"status": "saved"})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4444)
