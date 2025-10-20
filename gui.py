from __future__ import annotations

import re
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
PLAYLISTS_PATH = BASE_DIR / "playlists.yml"
LEGEND_PATH = BASE_DIR / "legend.txt"

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
        ("", "None"),
        ("popularity", "Spotify Popularity"),
        ("ratingCount", "Rating Count"),
        ("parentRatingCount", "Album Rating Count"),
        ("year", "Track Year"),
        ("parentYear", "Album Year"),
        ("originallyAvailableAt", "Release Date"),
    ]
)


def humanize_field_name(field: str) -> str:
    if field in FIELD_LABEL_OVERRIDES:
        return FIELD_LABEL_OVERRIDES[field]

    label = field.replace(".", " ").replace("_", " ")
    label = re.sub(r"(?<!^)(?=[A-Z])", " ", label)
    words = [w.upper() if w.lower() in {"id", "guid", "uuid"} else w for w in label.split()]
    humanized = " ".join(word.capitalize() if word.islower() else word for word in words)
    return humanized


def load_field_options() -> List[Dict[str, str]]:
    if not LEGEND_PATH.exists():
        # Fallback to a minimal set if the legend is unavailable
        fallback_fields = {
            "title",
            "artist",
            "album",
            "genres",
            "year",
            "parentYear",
            "grandparentTitle",
            "ratingCount",
        }
        return [
            {"value": field, "label": humanize_field_name(field)}
            for field in sorted(fallback_fields)
        ]

    fields: Dict[str, str] = {}
    with LEGEND_PATH.open("r", encoding="utf-8") as legend_file:
        for raw_line in legend_file:
            line = raw_line.strip()
            if not line or line.startswith("=") or line.startswith("-") or line.startswith("🔹"):
                continue
            if line.startswith(":"):
                continue
            line = line.split("(")[0].strip()
            if not line:
                continue
            field_name = line.split()[0]
            if not field_name:
                continue
            fields[field_name] = humanize_field_name(field_name)

    sorted_fields = sorted(fields.items(), key=lambda item: item[1].lower())
    return [{"value": value, "label": label} for value, label in sorted_fields]


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
            if key not in {"limit", "artist_limit", "sort_by", "plex_filter"}
        }
        playlists_data.append(
            {
                "name": name,
                "limit": config.get("limit", 0) or 0,
                "artist_limit": config.get("artist_limit", 0) or 0,
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
        sort_by = playlist_entry.get("sort_by") or None

        playlist_config: Dict[str, Any] = {}
        extras = playlist_entry.get("extras")
        if isinstance(extras, dict):
            playlist_config.update(extras)
        playlist_config["limit"] = max(limit, 0)
        playlist_config["artist_limit"] = max(artist_limit, 0)
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

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

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

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4444)
