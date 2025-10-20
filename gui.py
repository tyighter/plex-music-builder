from __future__ import annotations

import re
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Tuple

import yaml
from flask import Flask, jsonify, render_template, request


def _represent_ordered_dict(dumper: yaml.Dumper, data: OrderedDict) -> Any:
    """Ensure OrderedDict values can be serialized by ``yaml.safe_dump``."""

    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


yaml.SafeDumper.add_representer(OrderedDict, _represent_ordered_dict)

BASE_DIR = Path(__file__).resolve().parent
PLAYLISTS_PATH = BASE_DIR / "playlists.yml"
LEGEND_PATH = BASE_DIR / "legend.txt"
CONFIG_PATH = BASE_DIR / "config.yml"
DEFAULT_ALLMUSIC_CACHE = Path("/app/allmusic_popularity.json")

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
    "genres",
    "moods",
    "styles",
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


def _format_timestamp(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


class BuildManager:
    _INFO_LINE_PATTERN = re.compile(r"\[(?P<level>[A-Z]+)\]\s*(?P<message>.*)")
    _PLAYLIST_QUOTED_PATTERN = re.compile(r"'([^']+)'")

    def __init__(
        self,
        command: Optional[List[str]] = None,
        work_dir: Optional[Path] = None,
    ) -> None:
        self._command = command or [sys.executable, "main.py"]
        self._work_dir = str(work_dir) if work_dir is not None else None
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._last_exit_code: Optional[int] = None
        self._last_message: Optional[str] = None
        self._last_started_at: Optional[datetime] = None
        self._last_finished_at: Optional[datetime] = None
        self._active_job: Optional[Dict[str, Any]] = None
        self._last_all_result: Optional[Dict[str, Any]] = None
        self._last_playlist_results: Dict[str, Dict[str, Any]] = {}
        self._stop_requested = False
        self._log_thread: Optional[threading.Thread] = None
        self._playlist_logs: Dict[str, List[str]] = {}
        self._general_logs: List[str] = []

    @staticmethod
    def _normalize_playlist_key(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        key = str(name).strip()
        return key or None

    @classmethod
    def _extract_info_message(cls, line: str) -> Optional[str]:
        raw = line.strip()
        if not raw:
            return None

        match = cls._INFO_LINE_PATTERN.search(raw)
        if match:
            level = match.group("level")
            if level != "INFO":
                return None
            message = match.group("message").strip()
            return message

        if raw.startswith("INFO"):
            parts = raw.split(":", 1)
            if len(parts) == 2:
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
        logs.append(message)
        if len(logs) > 200:
            del logs[:-200]

    def _append_general_log_locked(self, message: str) -> None:
        self._general_logs.append(message)
        if len(self._general_logs) > 400:
            del self._general_logs[:-400]

    def _handle_log_line(self, line: str) -> None:
        info_message = self._extract_info_message(line)
        if info_message is None:
            return

        playlist_name = self._extract_playlist_from_message(info_message)
        with self._lock:
            if playlist_name:
                if info_message.lower().startswith("building playlist:"):
                    self._playlist_logs[playlist_name] = []
                self._append_playlist_log_locked(playlist_name, info_message)
            else:
                self._append_general_log_locked(info_message)

    def _consume_process_output(self, stream: IO[str]) -> None:
        try:
            for raw_line in stream:
                if not raw_line:
                    continue
                self._handle_log_line(raw_line)
        finally:
            try:
                stream.close()
            except Exception:  # pragma: no cover - defensive
                pass
            with self._lock:
                self._log_thread = None

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
        if self._process is None:
            return

        return_code = self._process.poll()
        if return_code is None:
            return

        job = self._active_job.copy() if self._active_job is not None else None
        started_at = self._last_started_at
        finished_at = _utcnow()
        self._last_exit_code = return_code
        self._process = None
        self._last_finished_at = finished_at

        if self._stop_requested:
            if job and job.get("type") == "playlist" and job.get("playlist"):
                self._last_message = (
                    self._last_message
                    or f"Build for playlist '{job['playlist']}' stopped."
                )
            elif job and job.get("type") == "all":
                self._last_message = (
                    self._last_message or "Build for all playlists stopped."
                )
            else:
                self._last_message = self._last_message or "Build process stopped."
            state = "stopped"
        else:
            if return_code == 0:
                self._last_message = self._last_message or "Build completed successfully."
                state = "success"
            else:
                self._last_message = (
                    self._last_message or f"Build exited with code {return_code}."
                )
                state = "error"

        self._record_job_result_locked(
            job,
            return_code,
            started_at,
            finished_at,
            state,
            self._last_message,
        )
        self._active_job = None
        self._stop_requested = False

    def _status_snapshot_locked(self) -> Dict[str, Any]:
        self._normalize_process_state_locked()

        running = self._process is not None
        status_label = "running" if running else "idle"
        if not running and self._last_exit_code not in (None, 0):
            status_label = "error"

        job = self._active_job.copy() if self._active_job is not None else None
        playlist_results = {
            name: result.copy()
            for name, result in self._last_playlist_results.items()
        }
        playlist_logs = {
            name: list(entries)
            for name, entries in self._playlist_logs.items()
        }

        return {
            "running": running,
            "status": status_label,
            "pid": self._process.pid if running and self._process is not None else None,
            "since": _format_timestamp(self._last_started_at),
            "last_finished": _format_timestamp(self._last_finished_at),
            "exit_code": self._last_exit_code,
            "message": self._last_message,
            "job": job,
            "results": {
                "all": self._last_all_result.copy() if self._last_all_result else None,
                "playlists": playlist_results,
            },
            "logs": {
                "playlists": playlist_logs,
                "general": list(self._general_logs),
            },
        }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return self._status_snapshot_locked()

    def start(self, playlist: Optional[str] = None) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        with self._lock:
            self._normalize_process_state_locked()
            if self._process is not None:
                message = "Builder is already running."
                self._last_message = message
                return False, self._status_snapshot_locked(), message

            playlist_name: Optional[str]
            if playlist is None:
                playlist_name = None
            else:
                playlist_name = str(playlist).strip() or None
                if playlist_name is None:
                    message = "Playlist name is required to start a build."
                    self._last_message = message
                    return False, self._status_snapshot_locked(), message

            command = list(self._command)
            job: Dict[str, Any]
            if playlist_name:
                command = command + ["--playlist", playlist_name]
                job = {"type": "playlist", "playlist": playlist_name}
                job_message = f"Build started for playlist '{playlist_name}'."
                self._playlist_logs.pop(playlist_name, None)
            else:
                job = {"type": "all"}
                job_message = "Build for all playlists started."
                self._playlist_logs = {}
                self._general_logs = []

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
                self._process = None
                self._last_message = error_message
                self._last_started_at = None
                self._last_finished_at = _utcnow()
                self._last_exit_code = None
                status = self._status_snapshot_locked()
                status["status"] = "error"
                return False, status, error_message

            self._process = process
            self._active_job = job
            self._last_exit_code = None
            self._last_finished_at = None
            self._last_started_at = _utcnow()
            self._last_message = job_message
            self._stop_requested = False
            if process.stdout is not None:
                self._log_thread = threading.Thread(
                    target=self._consume_process_output,
                    args=(process.stdout,),
                    name="build-log-consumer",
                    daemon=True,
                )
                self._log_thread.start()
            status = self._status_snapshot_locked()
            return True, status, job_message

    def stop(self, timeout: float = 10.0) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        with self._lock:
            self._normalize_process_state_locked()
            if self._process is None:
                message = "Builder is not running."
                self._last_message = message
                return False, self._status_snapshot_locked(), message

            process = self._process
            job = self._active_job.copy() if self._active_job is not None else None
            if job and job.get("type") == "playlist" and job.get("playlist"):
                stopping_message = f"Stopping build for playlist '{job['playlist']}'."
            elif job and job.get("type") == "all":
                stopping_message = "Stopping build for all playlists."
            else:
                stopping_message = "Stopping build."
            self._last_message = stopping_message
            self._stop_requested = True

        try:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                self._stop_requested = False
                error_message = f"Unable to stop build: {exc}"
                self._last_message = error_message
                status = self._status_snapshot_locked()
                status["status"] = "error"
                return False, status, error_message

        with self._lock:
            self._normalize_process_state_locked()
            message = self._last_message or "Build process stopped."
            status = self._status_snapshot_locked()
            return True, status, message


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

    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
                config_data = yaml.safe_load(config_file) or {}
        except Exception:
            config_data = {}

        allmusic_cfg = config_data.get("allmusic") or {}
        if isinstance(allmusic_cfg, dict):
            cache_file_raw = allmusic_cfg.get("cache_file")
            if isinstance(cache_file_raw, str) and cache_file_raw.strip():
                candidate = Path(cache_file_raw.strip())
                if not candidate.is_absolute():
                    candidate = (CONFIG_PATH.parent / candidate).resolve()
                cache_path = candidate

    return cache_path


def load_yaml_data() -> Dict[str, Any]:
    if not PLAYLISTS_PATH.exists():
        return {"defaults": {"plex_filter": []}, "playlists": OrderedDict()}

    with PLAYLISTS_PATH.open("r", encoding="utf-8") as playlist_file:
        data = yaml.safe_load(playlist_file) or {}

    defaults = data.get("defaults", {}) or {}
    playlists = data.get("playlists", {}) or {}
    return {"defaults": defaults, "playlists": playlists}


def normalize_filter_entry(filter_entry: Dict[str, Any]) -> Dict[str, Any]:
    field = filter_entry.get("field", "")
    operator = filter_entry.get("operator", "equals")
    value = filter_entry.get("value", "")
    match_all = filter_entry.get("match_all")

    if isinstance(value, list):
        value_str = ", ".join(str(item) for item in value)
    else:
        value_str = "" if value is None else str(value)

    return {
        "field": field,
        "operator": operator,
        "value": value_str,
        "match_all": bool(match_all) if match_all is not None else True,
    }


def serialize_filters(filters: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not filters:
        return []
    return [normalize_filter_entry(filter_entry or {}) for filter_entry in filters]


def load_playlists() -> Dict[str, Any]:
    data = load_yaml_data()
    defaults_config = data.get("defaults", {}) or {}
    defaults_filters = serialize_filters(defaults_config.get("plex_filter"))
    defaults_extras = {
        key: value
        for key, value in defaults_config.items()
        if key != "plex_filter"
    }

    playlists_data = []
    for name, config in data.get("playlists", {}).items():
        config = config or {}
        extras = {
            key: value
            for key, value in config.items()
            if key not in {"limit", "artist_limit", "album_limit", "sort_by", "plex_filter"}
        }
        playlists_data.append(
            {
                "name": name,
                "limit": config.get("limit", 0) or 0,
                "artist_limit": config.get("artist_limit", 0) or 0,
                "album_limit": config.get("album_limit", 0) or 0,
                "sort_by": config.get("sort_by", ""),
                "plex_filter": serialize_filters(config.get("plex_filter")),
                "extras": extras,
            }
        )

    return {
        "defaults": {"plex_filter": defaults_filters, "extras": defaults_extras},
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

    return yaml_entry


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def save_playlists(payload: Dict[str, Any]) -> None:
    defaults_payload = payload.get("defaults", {}) or {}
    playlists_payload = payload.get("playlists", []) or []

    defaults_filters = []
    for filter_entry in defaults_payload.get("plex_filter", []):
        yaml_filter = build_filter_for_yaml(filter_entry)
        if yaml_filter is not None:
            defaults_filters.append(yaml_filter)

    defaults_config: Dict[str, Any] = {}
    extras = defaults_payload.get("extras")
    if isinstance(extras, dict):
        defaults_config.update(extras)
    if defaults_filters:
        defaults_config["plex_filter"] = defaults_filters

    playlists_dict: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for playlist_entry in playlists_payload:
        name = (playlist_entry.get("name") or "").strip()
        if not name:
            continue

        limit = to_int(playlist_entry.get("limit", 0))
        artist_limit = to_int(playlist_entry.get("artist_limit", 0))
        album_limit = to_int(playlist_entry.get("album_limit", 0))
        sort_by = playlist_entry.get("sort_by") or None

        playlist_config: Dict[str, Any] = {}
        extras = playlist_entry.get("extras")
        if isinstance(extras, dict):
            playlist_config.update(extras)
        playlist_config["limit"] = max(limit, 0)
        playlist_config["artist_limit"] = max(artist_limit, 0)
        playlist_config["album_limit"] = max(album_limit, 0)
        if sort_by:
            playlist_config["sort_by"] = sort_by

        playlist_filters = []
        for filter_entry in playlist_entry.get("plex_filter", []):
            yaml_filter = build_filter_for_yaml(filter_entry)
            if yaml_filter is not None:
                playlist_filters.append(yaml_filter)
        if playlist_filters:
            playlist_config["plex_filter"] = playlist_filters

        playlists_dict[name] = playlist_config

    yaml_structure: Dict[str, Any] = {}
    yaml_structure["defaults"] = defaults_config
    yaml_structure["playlists"] = playlists_dict

    with PLAYLISTS_PATH.open("w", encoding="utf-8") as playlist_file:
        yaml.safe_dump(yaml_structure, playlist_file, sort_keys=False, allow_unicode=True)


def _determine_separator(raw_path: str) -> str:
    if raw_path.count("\\") > raw_path.count("/"):
        return "\\"
    return "/"


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

        for candidate in candidates[:50]:
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
                }
            )

        return jsonify({
            "directory": str(directory),
            "entries": entries,
        })

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

    @app.route("/api/cache/allmusic", methods=["POST"])
    def clear_allmusic_cache() -> Any:
        cache_path = resolve_allmusic_cache_path()

        try:
            if cache_path.exists():
                cache_path.unlink()
                return jsonify({
                    "status": "cleared",
                    "path": str(cache_path),
                })
        except Exception as exc:
            return jsonify({"error": f"Unable to clear AllMusic cache: {exc}"}), 500

        return jsonify({
            "status": "missing",
            "path": str(cache_path),
        })

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4444)
