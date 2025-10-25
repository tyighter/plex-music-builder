import argparse
import os
import re
import sys
import yaml
import math
import logging
from logging.handlers import TimedRotatingFileHandler
import requests
import json
import time
import threading
import copy
import datetime
from pathlib import Path
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cmp_to_key, lru_cache
from xml.etree import ElementTree as ET
from html import unescape
import unicodedata
import ast
from urllib.parse import unquote, quote, urlparse

from plexapi.server import PlexServer
from tqdm import tqdm

# ----------------------------
# Load Config and Setup Logging
# ----------------------------
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG_SEARCH_PATHS = []


def _resolve_config_path() -> Path:
    """Return the most appropriate config file path for the current run."""

    candidates = []

    env_override = os.environ.get("PMB_CONFIG_PATH")
    if env_override:
        candidates.append(Path(env_override).expanduser())

    candidates.append(Path("/app/config.yml"))
    candidates.append(Path(__file__).resolve().parent / "config.yml")

    CONFIG_SEARCH_PATHS[:] = candidates

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


CONFIG_PATH = _resolve_config_path()

if not CONFIG_PATH.exists():
    searched = ", ".join(str(path) for path in CONFIG_SEARCH_PATHS)
    raise FileNotFoundError(
        "Unable to locate configuration file. Checked paths: " + searched
    )

CONFIG_PATH = CONFIG_PATH.resolve()


def _resolve_runtime_dir(config_dir: Path) -> Path:
    """Determine where runtime artefacts (logs, caches) should live."""

    env_override = os.environ.get("PMB_RUNTIME_DIR")
    if env_override:
        return Path(env_override).expanduser()

    app_dir = Path("/app")
    if app_dir.exists():
        return app_dir

    return config_dir


def _resolve_path_setting(raw_value, default_path: Path, base_dir: Path) -> str:
    """Resolve a path from config, allowing relative paths."""

    if isinstance(raw_value, str) and raw_value.strip():
        candidate = Path(raw_value.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
    else:
        candidate = default_path

    return str(candidate)


cfg = load_yaml(CONFIG_PATH)

CONFIG_DIR = CONFIG_PATH.parent
RUNTIME_DIR = _resolve_runtime_dir(CONFIG_DIR).resolve()

PLAYLISTS_FILE = str((CONFIG_DIR / "playlists.yml").resolve())
PLAYLISTS_BASE_DIR = os.path.dirname(PLAYLISTS_FILE) or "."

# Backward compatible config loading
if isinstance(cfg.get("plex"), dict):
    PLEX_URL = cfg["plex"].get("PLEX_URL")
    PLEX_TOKEN = cfg["plex"].get("PLEX_TOKEN")
    LIBRARY_NAME = cfg["plex"].get("library_name", "Music")
else:
    PLEX_URL = cfg.get("PLEX_URL")
    PLEX_TOKEN = cfg.get("PLEX_TOKEN")
    LIBRARY_NAME = cfg.get("library_name", "Music")

# Runtime control
runtime_cfg = cfg.get("runtime", {})
RUN_FOREVER = runtime_cfg.get("run_forever", False)
CACHE_ONLY = runtime_cfg.get("cache_only", False)
REFRESH_INTERVAL = runtime_cfg.get("refresh_interval_minutes", 60)
SAVE_INTERVAL = runtime_cfg.get("save_interval", 100)  # save cache every N items
PLAYLIST_CHUNK_SIZE = runtime_cfg.get("playlist_chunk_size", 200)
MAX_WORKERS = runtime_cfg.get("max_workers", 3)

def _coerce_positive_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


def _coerce_positive_int(value):
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


allmusic_cfg = cfg.get("allmusic", {}) or {}
ALLMUSIC_ENABLED = allmusic_cfg.get("enabled", True)
ALLMUSIC_CACHE_FILE = _resolve_path_setting(
    allmusic_cfg.get("cache_file"),
    RUNTIME_DIR / "allmusic_popularity.json",
    CONFIG_DIR,
)
ALLMUSIC_TIMEOUT = allmusic_cfg.get("timeout", 10)
ALLMUSIC_USER_AGENT = allmusic_cfg.get(
    "user_agent",
    "plex-music-builder/1.0 (+https://github.com/plexmusicbuilder)",
)
ALLMUSIC_GOOGLE_MIN_INTERVAL = float(allmusic_cfg.get("google_min_interval", 2.0))
ALLMUSIC_GOOGLE_BACKOFF = float(allmusic_cfg.get("google_backoff_seconds", 30.0))
ALLMUSIC_CACHE_VERSION = 3

logging_cfg = cfg.get("logging", {})
LOG_LEVEL = logging_cfg.get("level", "DEBUG").upper()
LOG_FILE = _resolve_path_setting(
    logging_cfg.get("file"),
    RUNTIME_DIR / "logs/plex_music_builder.log",
    CONFIG_DIR,
)
ACTIVE_LOG_FILE = None

_default_log_dir = os.path.dirname(LOG_FILE) if LOG_FILE else ""
if not _default_log_dir:
    _default_log_dir = str((RUNTIME_DIR / "logs").resolve())

PLAYLIST_LOG_DIR = logging_cfg.get("playlist_debug_dir")
if isinstance(PLAYLIST_LOG_DIR, str) and not PLAYLIST_LOG_DIR.strip():
    PLAYLIST_LOG_DIR = None
if PLAYLIST_LOG_DIR is None:
    PLAYLIST_LOG_DIR = os.path.join(_default_log_dir, "playlists")

if not PLEX_URL or not PLEX_TOKEN:
    raise EnvironmentError("PLEX_URL and PLEX_TOKEN must be set in config.yml")


def setup_logging():
    """Configure logging to stream to stdout and a persistent log file."""
    logger_name = "plex_music_builder"
    logger_obj = logging.getLogger(logger_name)

    # Avoid duplicating handlers if setup_logging is called multiple times
    if logger_obj.handlers:
        return logger_obj

    global ACTIVE_LOG_FILE

    log_level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger_obj.setLevel(log_level)
    logger_obj.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger_obj.addHandler(stream_handler)

    if LOG_FILE:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        truncate_error = False
        try:
            with open(LOG_FILE, "w", encoding="utf-8"):
                pass
        except OSError as exc:
            truncate_error = True
            logger_obj.error(
                f"Unable to initialise log file '{LOG_FILE}' for writing: {exc}"
            )

        if not truncate_error:
            try:
                file_handler = TimedRotatingFileHandler(
                    LOG_FILE,
                    when="midnight",
                    backupCount=7,
                    encoding="utf-8",
                    utc=False,
                    delay=False,
                    interval=1,
                )
            except OSError as exc:
                logger_obj.error(f"Unable to open log file '{LOG_FILE}': {exc}")
            else:
                file_handler.setFormatter(formatter)
                file_handler.setLevel(log_level)
                logger_obj.addHandler(file_handler)
                ACTIVE_LOG_FILE = LOG_FILE

    return logger_obj


logger = setup_logging()
if ACTIVE_LOG_FILE:
    logger.info(f"Detailed logs will be written to: {ACTIVE_LOG_FILE}")


_thread_local_logger = threading.local()


def _get_active_logger():
    """Return the logger associated with the current thread, if any."""

    thread_logger = getattr(_thread_local_logger, "current", None)
    if thread_logger is not None:
        return thread_logger
    return logger


# Mapping of user-friendly filter fields to the Plex attribute names
# exposed on track metadata objects. These aliases line up with the
# guidance documented in ``legend.txt`` so that filters like
# ``field: artist`` transparently evaluate the underlying
# ``grandparentTitle`` value returned by Plex.
FIELD_ALIASES = {
    # Artist level shortcuts
    "artist": "grandparentTitle",
    "artist.title": "grandparentTitle",
    "artist.id": "grandparentRatingKey",
    "artist.ratingKey": "grandparentRatingKey",
    "artist.guid": "grandparentGuid",
    "artist.thumb": "grandparentThumb",
    "artist.art": "grandparentArt",
    # Album level shortcuts
    "album": "parentTitle",
    "album.title": "parentTitle",
    "album.id": "parentRatingKey",
    "album.ratingKey": "parentRatingKey",
    "album.guid": "parentGuid",
    "album.thumb": "parentThumb",
    "album.art": "parentArt",
    "album.year": "parentYear",
    # Track level shortcuts
    "title": "title",
    "track": "title",
    "track.title": "title",
}


class PlaylistThreadFilter(logging.Filter):
    """Filter log records to a specific thread and inject playlist metadata."""

    def __init__(self, playlist_name, thread_id):
        super().__init__()
        self.playlist_name = playlist_name
        self.thread_id = thread_id

    def filter(self, record):
        if record.thread != self.thread_id:
            return False
        record.playlist = self.playlist_name
        return True


class PlaylistLoggerProxy:
    """Proxy logger that always emits DEBUG messages to a playlist handler."""

    def __init__(self, base_logger, playlist_handler):
        self._base_logger = base_logger
        self._playlist_handler = playlist_handler

    def __getattr__(self, name):
        return getattr(self._base_logger, name)

    def isEnabledFor(self, level):
        if level == logging.DEBUG and self._playlist_handler:
            return True
        return self._base_logger.isEnabledFor(level)

    def _emit_playlist_record(self, level, msg, args, kwargs):
        if not self._playlist_handler:
            return

        stacklevel = kwargs.get("stacklevel", 1) + 1
        stack_info = kwargs.get("stack_info", False)
        fn, lno, func, sinfo = self._base_logger.findCaller(
            stack_info=stack_info,
            stacklevel=stacklevel,
        )
        record = self._base_logger.makeRecord(
            self._base_logger.name,
            level,
            fn,
            lno,
            msg,
            args,
            kwargs.get("exc_info"),
            func,
            kwargs.get("extra"),
            sinfo,
        )
        self._playlist_handler.handle(record)

    def debug(self, msg, *args, **kwargs):
        self._emit_playlist_record(logging.DEBUG, msg, args, kwargs)
        if self._base_logger.isEnabledFor(logging.DEBUG):
            self._base_logger.debug(msg, *args, **kwargs)

def _sanitize_playlist_name(name):
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return sanitized or "playlist"


def _create_playlist_log_handler(playlist_name):
    """Create a log handler that captures logs for a specific playlist."""

    if not PLAYLIST_LOG_DIR:
        return None, None

    os.makedirs(PLAYLIST_LOG_DIR, exist_ok=True)

    thread_id = threading.get_ident()
    safe_name = _sanitize_playlist_name(playlist_name)
    filename = f"{safe_name}.debug.log"
    filepath = os.path.join(PLAYLIST_LOG_DIR, filename)

    handler = logging.FileHandler(filepath, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(playlist)s | %(message)s")
    handler.setFormatter(formatter)
    handler.addFilter(PlaylistThreadFilter(playlist_name, thread_id))

    return handler, filepath

# ----------------------------
# Connect to Plex
# ----------------------------
plex = None


def get_plex_server():
    """Return a cached ``PlexServer`` instance.

    The CLI argument parsing happens near the bottom of this module. Import
    side-effects that immediately talk to Plex make even ``--help`` fail when
    credentials point to an unavailable server. Lazily initialising the client
    avoids that situation while keeping backwards compatibility for code that
    still relies on the ``main.plex`` attribute.
    """

    global plex
    if plex is None:
        plex = PlexServer(PLEX_URL, PLEX_TOKEN)
    return plex

# ----------------------------
# Load Playlists Definition
# ----------------------------
raw_playlists = load_yaml(PLAYLISTS_FILE)


def _normalize_playlist_source(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"plex", "spotify"}:
            return normalized
    return "plex"


def _merge_playlist_defaults(raw_payload):
    if not isinstance(raw_payload, dict):
        return {}

    defaults = raw_payload.get("defaults")
    playlists = raw_payload.get("playlists")

    if not isinstance(playlists, dict):
        return {}

    default_cfg = copy.deepcopy(defaults) if isinstance(defaults, dict) else {}
    default_filters = default_cfg.pop("plex_filter", []) or []
    default_boosts = default_cfg.pop("popularity_boosts", []) or []

    merged_playlists = {}
    for playlist_name, playlist_cfg in playlists.items():
        cfg = copy.deepcopy(playlist_cfg) if isinstance(playlist_cfg, dict) else {}

        combined = copy.deepcopy(default_cfg)
        combined.update(cfg)

        playlist_source = _normalize_playlist_source(cfg.get("source"))

        playlist_filters = cfg.get("plex_filter", []) or []
        playlist_boosts = cfg.get("popularity_boosts", []) or []
        combined_filters = []
        if playlist_source != "spotify":
            if default_filters:
                combined_filters.extend(copy.deepcopy(default_filters))
            if playlist_filters:
                combined_filters.extend(copy.deepcopy(playlist_filters))

        if combined_filters:
            combined["plex_filter"] = combined_filters
        else:
            combined.pop("plex_filter", None)

        combined_boosts = []
        if playlist_source != "spotify":
            if default_boosts:
                combined_boosts.extend(copy.deepcopy(default_boosts))
            if playlist_boosts:
                combined_boosts.extend(copy.deepcopy(playlist_boosts))

        if combined_boosts:
            combined["popularity_boosts"] = combined_boosts
        else:
            combined.pop("popularity_boosts", None)

        merged_playlists[playlist_name] = combined

    return merged_playlists


playlists_data = _merge_playlist_defaults(raw_playlists)

# ----------------------------
# Metadata Cache
# ----------------------------
CACHE_FILE = str((RUNTIME_DIR / "metadata_cache.json").resolve())
METADATA_PROVIDER_URL = "https://metadata.provider.plex.tv"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        try:
            metadata_cache = json.load(f)
        except json.JSONDecodeError:
            metadata_cache = {}
else:
    metadata_cache = {}

# In-memory caches used during filtering. These dramatically reduce repeated
# metadata parsing and field aggregation work when evaluating large playlists.
_METADATA_FIELD_CACHE = OrderedDict()
_METADATA_FIELD_CACHE_MAX_SIZE = 50000

_TRACK_FIELD_CACHE = OrderedDict()
_TRACK_FIELD_CACHE_MAX_SIZE = 20000


_CACHE_LOOKUP_SENTINEL = object()
_ALBUM_YEAR_CACHE = {}
_ALBUM_YEAR_MISS_KEYS = set()
_ALBUM_GUID_CACHE = {}
_ALBUM_GUID_MISS_KEYS = set()
_TRACK_GUID_CACHE = {}
_TRACK_GUID_MISS_KEYS = set()


def _lookup_cached_value(cache, misses, key):
    if key is None:
        return _CACHE_LOOKUP_SENTINEL

    key_str = str(key)
    if key_str in cache:
        return cache[key_str]
    if key_str in misses:
        return None

    return _CACHE_LOOKUP_SENTINEL


def _store_cached_value(cache, misses, key, value):
    if key is None:
        return

    key_str = str(key)
    if value is None:
        misses.add(key_str)
        cache.pop(key_str, None)
    else:
        cache[key_str] = value
        misses.discard(key_str)


def _touch_cache_entry(cache: OrderedDict, key):
    if key in cache:
        cache.move_to_end(key)


def _set_cache_entry(cache: OrderedDict, key, value, max_size: int):
    cache[key] = value
    cache.move_to_end(key)
    if max_size and len(cache) > max_size:
        cache.popitem(last=False)


def _build_track_cache_key(track):
    if track is None:
        return None
    rating_key = getattr(track, "ratingKey", None)
    if rating_key is not None:
        return f"track:{rating_key}"
    return f"object:{id(track)}"

metadata_cache_lock = threading.Lock()
metadata_cache_dirty = False
_metadata_cache_inserts_since_save = 0


def _write_metadata_cache_locked():
    cache_dir = os.path.dirname(CACHE_FILE)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_cache, f)


def _mark_metadata_cache_dirty_locked():
    global metadata_cache_dirty, _metadata_cache_inserts_since_save

    metadata_cache_dirty = True
    _metadata_cache_inserts_since_save += 1

    if SAVE_INTERVAL and SAVE_INTERVAL > 0 and _metadata_cache_inserts_since_save >= SAVE_INTERVAL:
        _write_metadata_cache_locked()
        metadata_cache_dirty = False
        _metadata_cache_inserts_since_save = 0


def flush_metadata_cache(force=False):
    """Persist the metadata cache to disk if new entries were added."""

    global metadata_cache_dirty, _metadata_cache_inserts_since_save

    with metadata_cache_lock:
        if not metadata_cache_dirty and not force:
            return

        _write_metadata_cache_locked()
        metadata_cache_dirty = False
        _metadata_cache_inserts_since_save = 0


def save_cache():
    flush_metadata_cache(force=True)
def _strip_parenthetical(value):
    if not value:
        return ""
    return re.sub(r"\s*\([^)]*\)\s*", " ", str(value)).strip()


def _normalize_compare_value(value):
    if not value:
        return ""
    normalized = re.sub(r"\s+", " ", str(value)).strip().lower()
    return normalized


_SPOTIFY_PLAYLIST_ID_RE = re.compile(
    r"(?:https?://open\.spotify\.com/playlist/|spotify:playlist:)([A-Za-z0-9]+)"
)
_SPOTIFY_ENTITY_PATTERN = re.compile(r"Spotify\.Entity\s*=\s*({.*?})\s*;", re.DOTALL)
_SPOTIFY_ENTITY_JSON_PARSE_PATTERN = re.compile(
    r"Spotify\.Entity\s*=\s*JSON\.parse\(\s*(?P<decoder>decodeURIComponent\()?(?P<quoted>\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*')\s*\)?\s*\)\s*;",
    re.DOTALL,
)
_SPOTIFY_NEXT_DATA_PATTERN = re.compile(
    r"<script[^>]+id=\"__NEXT_DATA__\"[^>]*>(?P<payload>{.*?})</script>",
    re.DOTALL,
)
_SPOTIFY_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _extract_spotify_playlist_id(raw_url):
    if not raw_url:
        return ""

    candidate = str(raw_url).strip()
    match = _SPOTIFY_PLAYLIST_ID_RE.search(candidate)
    if match:
        return match.group(1)

    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https"} and parsed.netloc.endswith("spotify.com"):
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] == "playlist":
            return path_parts[1]

    return ""


def _normalize_spotify_playlist_url(raw_url):
    playlist_id = _extract_spotify_playlist_id(raw_url)
    if not playlist_id:
        raise ValueError(f"Invalid Spotify playlist URL: {raw_url!r}")
    return f"https://open.spotify.com/playlist/{playlist_id}"


def _collect_tracks_from_next_data(root_node):
    tracks = []
    seen_keys = set()

    def _extract_key(info):
        uri = info.get("uri")
        if uri:
            return ("uri", uri)
        return (
            "identity",
            _normalize_compare_value(info.get("name")),
            _normalize_compare_value(info.get("album")),
            _normalize_compare_value(info.get("artist")),
        )

    def _append_tracks(track_infos):
        for info in track_infos:
            key = _extract_key(info)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            album_artists = []
            primary_artist = info.get("artist") or ""
            if primary_artist:
                album_artists.append({"name": primary_artist})
            album_info = {}
            album_name = info.get("album") or ""
            if album_name or album_artists:
                album_info = {"name": album_name or ""}
                if album_artists:
                    album_info["artists"] = album_artists
            track_payload = {
                "name": info.get("name") or "",
                "album": album_info,
                "artists": album_artists,
            }
            uri = info.get("uri")
            if uri:
                track_payload["uri"] = uri
            tracks.append({"track": track_payload})

    def _walk(node):
        if isinstance(node, list):
            converted = []
            fallback_nodes = []
            for item in node:
                track_info = _coerce_next_data_track(item)
                if track_info:
                    converted.append(track_info)
                else:
                    fallback_nodes.append(item)
            if converted:
                _append_tracks(converted)
            for item in fallback_nodes:
                _walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                _walk(value)

    _walk(root_node)
    return tracks


def _coerce_next_data_track(node):
    if isinstance(node, dict):
        # unwrap common containers used in the Next.js payload
        for key in ("track", "item", "itemV2", "itemV1", "data"):
            value = node.get(key)
            if isinstance(value, (dict, list)):
                track_info = _coerce_next_data_track(value)
                if track_info:
                    return track_info

        if "trackMetadata" in node and isinstance(node["trackMetadata"], dict):
            metadata = node["trackMetadata"]
            return _build_next_data_track_info(
                uri=metadata.get("trackUri"),
                name=metadata.get("trackName"),
                album=metadata.get("albumName"),
                artist=metadata.get("artistName"),
            )

        typename = node.get("__typename")
        if typename and typename not in {"Track", "MusicTrack", "Episode"}:
            # Ignore nodes that explicitly describe other entity types
            pass

        name = node.get("name") or node.get("trackName") or node.get("title")
        if not name and isinstance(node.get("metadata"), dict):
            metadata = node["metadata"]
            name = metadata.get("name") or metadata.get("trackName")

        if not name and isinstance(node.get("content"), dict):
            content = node["content"]
            name = content.get("title") or content.get("name")

        album_name = _extract_album_name_from_next_data(node)
        artist_name = _extract_artist_name_from_next_data(node)

        if not artist_name and isinstance(node.get("subtitle"), str):
            subtitle_parts = [part.strip() for part in node["subtitle"].split("•")]
            if subtitle_parts:
                artist_name = artist_name or subtitle_parts[0]
            if len(subtitle_parts) >= 2 and not album_name:
                album_name = subtitle_parts[1]

        uri = None
        for key in ("uri", "trackUri", "uriRaw"):
            value = node.get(key)
            if isinstance(value, str) and value.startswith("spotify:track:"):
                uri = value
                break

        if name and (artist_name or album_name or uri):
            return _build_next_data_track_info(
                uri=uri,
                name=name,
                album=album_name,
                artist=artist_name,
            )

    elif isinstance(node, list):
        converted = []
        for item in node:
            track_info = _coerce_next_data_track(item)
            if track_info:
                converted.append(track_info)
        if converted:
            return converted[0]

    return None


def _build_next_data_track_info(uri=None, name=None, album=None, artist=None):
    if not name:
        return None
    info = {
        "name": name,
        "album": album or "",
        "artist": artist or "",
    }
    if uri:
        info["uri"] = uri
    return info


def _extract_album_name_from_next_data(node):
    if isinstance(node, dict):
        if isinstance(node.get("album"), dict):
            album_name = node["album"].get("name") or node["album"].get("title")
            if album_name:
                return album_name
        if isinstance(node.get("albumOfTrack"), dict):
            album = node["albumOfTrack"]
            album_name = album.get("name") or album.get("title")
            if album_name:
                return album_name
            if isinstance(album.get("items"), list):
                for item in album["items"]:
                    album_name = _extract_album_name_from_next_data(item)
                    if album_name:
                        return album_name
        if isinstance(node.get("albumOfEpisode"), dict):
            album = node["albumOfEpisode"]
            album_name = album.get("name") or album.get("title")
            if album_name:
                return album_name
        if isinstance(node.get("album"), str):
            return node["album"]
        if isinstance(node.get("albumName"), str):
            return node["albumName"]
        if isinstance(node.get("trackMetadata"), dict):
            album_name = node["trackMetadata"].get("albumName")
            if album_name:
                return album_name
    return ""


def _extract_artist_name_from_next_data(node):
    if isinstance(node, dict):
        if isinstance(node.get("albumOfTrack"), dict):
            name = _extract_artist_name_from_next_data(node["albumOfTrack"])
            if name:
                return name
        if isinstance(node.get("albumOfEpisode"), dict):
            name = _extract_artist_name_from_next_data(node["albumOfEpisode"])
            if name:
                return name
        if isinstance(node.get("artists"), list):
            for entry in node["artists"]:
                name = _extract_artist_name_from_next_data(entry)
                if name:
                    return name
        if isinstance(node.get("artists"), dict):
            artists = node["artists"]
            name = artists.get("name") or artists.get("title")
            if name:
                return name
            if isinstance(artists.get("items"), list):
                for item in artists["items"]:
                    name = _extract_artist_name_from_next_data(item)
                    if name:
                        return name
        if isinstance(node.get("artistsOfTrack"), dict):
            artists_of_track = node["artistsOfTrack"]
            if isinstance(artists_of_track.get("items"), list):
                for item in artists_of_track["items"]:
                    name = _extract_artist_name_from_next_data(item)
                    if name:
                        return name
        if isinstance(node.get("profile"), dict):
            profile = node["profile"]
            name = profile.get("name") or profile.get("displayName")
            if name:
                return name
        if isinstance(node.get("artist"), dict):
            name = node["artist"].get("name") or node["artist"].get("title")
            if name:
                return name
        if isinstance(node.get("artist"), str):
            return node["artist"]
        if isinstance(node.get("artistName"), str):
            return node["artistName"]
        if isinstance(node.get("subtitle"), str):
            subtitle_parts = [part.strip() for part in node["subtitle"].split("•")]
            if subtitle_parts:
                return subtitle_parts[0]
    elif isinstance(node, list):
        for entry in node:
            name = _extract_artist_name_from_next_data(entry)
            if name:
                return name
    elif isinstance(node, str):
        return node
    return ""


def _extract_spotify_entity_payload(html_text):
    if not html_text:
        raise ValueError("Spotify playlist page was empty")

    match = _SPOTIFY_ENTITY_PATTERN.search(html_text)
    if not match:
        json_parse_match = _SPOTIFY_ENTITY_JSON_PARSE_PATTERN.search(html_text)
        if not json_parse_match:
            next_data_payload = _extract_spotify_entity_payload_from_next_data(html_text)
            if next_data_payload is None:
                raise ValueError("Unable to locate Spotify playlist metadata in page contents")
            return next_data_payload

        quoted_payload = json_parse_match.group("quoted")
        try:
            string_payload = ast.literal_eval(quoted_payload)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("Failed to parse Spotify playlist metadata") from exc

        if json_parse_match.group("decoder"):
            string_payload = unquote(string_payload)

        payload = string_payload
    else:
        payload = match.group(1)

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Failed to parse Spotify playlist metadata") from exc


def _extract_spotify_entity_payload_from_next_data(html_text):
    match = _SPOTIFY_NEXT_DATA_PATTERN.search(html_text)
    if not match:
        return None

    payload_text = match.group("payload")
    try:
        data = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Failed to parse Spotify playlist metadata") from exc

    track_items = _collect_tracks_from_next_data(data)
    if not track_items:
        return None

    return {"tracks": {"items": track_items}}


def _parse_spotify_entity_tracks(entity_payload):
    tracks_section = {}
    if isinstance(entity_payload, dict):
        tracks_section = entity_payload.get("tracks", {}) or {}

    track_items = tracks_section.get("items") if isinstance(tracks_section, dict) else None
    if not isinstance(track_items, list):
        return []

    parsed_tracks = []
    for item in track_items:
        track_info = item.get("track") if isinstance(item, dict) else None
        if not isinstance(track_info, dict):
            continue

        if track_info.get("is_local"):
            continue

        album_info = track_info.get("album") or {}
        artists = track_info.get("artists") or album_info.get("artists") or []
        primary_artist = ""
        if isinstance(artists, list) and artists:
            artist_entry = artists[0]
            if isinstance(artist_entry, dict):
                primary_artist = artist_entry.get("name") or ""

        parsed_tracks.append(
            {
                "title": track_info.get("name") or "",
                "album": album_info.get("name") or "",
                "artist": primary_artist,
            }
        )

    return parsed_tracks


def _match_spotify_tracks_to_library(spotify_tracks, library, log):
    matched_tracks = []
    unmatched_count = 0

    for position, spotify_track in enumerate(spotify_tracks, 1):
        raw_title = spotify_track.get("title") or ""
        raw_album = spotify_track.get("album") or ""
        raw_artist = spotify_track.get("artist") or ""

        title = _strip_parenthetical(raw_title)
        album = _strip_parenthetical(raw_album)
        artist = _strip_parenthetical(raw_artist)

        normalized_title = _normalize_compare_value(title)
        normalized_album = _normalize_compare_value(album)
        normalized_artist = _normalize_compare_value(artist)

        search_attempts = [
            {"title": title, "album": album, "artist": artist},
            {"title": title, "artist": artist},
            {"title": title, "album": album},
            {"title": title},
        ]

        search_results = []
        for params in search_attempts:
            query = {key: value for key, value in params.items() if value}
            if not query:
                continue

            try:
                search_results = library.searchTracks(**query)
            except Exception as exc:  # pragma: no cover - defensive logging
                log.warning(
                    "Spotify search failed for '%s' by '%s' (query=%s): %s",
                    raw_title or "<unknown>",
                    raw_artist or "<unknown>",
                    query,
                    exc,
                )
                search_results = []
            if search_results:
                break

        if not search_results:
            unmatched_count += 1
            if log.isEnabledFor(logging.INFO):
                log.info(
                    "Spotify track '%s' by '%s' on '%s' was not found in Plex",
                    raw_title or "<unknown>",
                    raw_artist or "<unknown>",
                    raw_album or "<unknown>",
                )
            continue

        def _candidate_score(candidate):
            candidate_title = _normalize_compare_value(
                _strip_parenthetical(getattr(candidate, "title", ""))
            )
            candidate_album = _normalize_compare_value(
                _strip_parenthetical(getattr(candidate, "parentTitle", ""))
            )
            candidate_artist = _normalize_compare_value(
                getattr(candidate, "grandparentTitle", "")
            )
            popularity = _resolve_track_popularity_value(candidate, playlist_logger=log)
            popularity_score = popularity if popularity is not None else -1

            return (
                1 if normalized_title and candidate_title == normalized_title else 0,
                1 if normalized_album and candidate_album == normalized_album else 0,
                1 if normalized_artist and candidate_artist == normalized_artist else 0,
                popularity_score,
            )

        best_candidate = max(search_results, key=_candidate_score)
        matched_tracks.append(best_candidate)

    return matched_tracks, unmatched_count


def _summarize_spotify_response(response, preview_limit=200):
    """Return a compact string describing a Spotify HTTP response."""

    status_code = getattr(response, "status_code", "<unknown>")
    content_length = "<unknown>"
    try:
        content = response.content
    except Exception:  # pragma: no cover - defensive fallback
        content = b""
    else:
        if content is not None:
            content_length = len(content)

    content_type = None
    try:
        content_type = response.headers.get("Content-Type")
    except Exception:  # pragma: no cover - headers behave like dict normally
        content_type = None

    preview = ""
    if content:
        try:
            text_preview = response.text
        except Exception:  # pragma: no cover - rare decode issue
            text_preview = ""
        if text_preview:
            preview = re.sub(r"\s+", " ", text_preview).strip()
            if len(preview) > preview_limit:
                preview = preview[:preview_limit] + "..."

    parts = [f"status={status_code}"]
    if content_type:
        parts.append(f"content_type={content_type}")
    parts.append(f"bytes={content_length}")
    if preview:
        parts.append(f"preview={preview}")

    return ", ".join(parts)


def _collect_spotify_tracks(spotify_url, library, log):
    normalized_url = _normalize_spotify_playlist_url(spotify_url)

    if log.isEnabledFor(logging.DEBUG):
        if spotify_url == normalized_url:
            log.debug("Fetching Spotify playlist from %s", normalized_url)
        else:
            log.debug(
                "Fetching Spotify playlist from %s (normalized from %s)",
                normalized_url,
                spotify_url,
            )

    try:
        response = requests.get(
            normalized_url, headers=_SPOTIFY_REQUEST_HEADERS, timeout=30
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Spotify request failed for {normalized_url}: {exc}"
        ) from exc

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "Spotify response for %s: %s",
            normalized_url,
            _summarize_spotify_response(response),
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            "Spotify request returned an error for %s: %s"
            % (normalized_url, _summarize_spotify_response(response))
        ) from exc

    try:
        entity_payload = _extract_spotify_entity_payload(response.text)
    except Exception as exc:
        raise RuntimeError(
            "Failed to parse Spotify playlist metadata from %s: %s (response %s)"
            % (normalized_url, exc, _summarize_spotify_response(response))
        ) from exc

    spotify_tracks = _parse_spotify_entity_tracks(entity_payload)
    matched_tracks, unmatched_count = _match_spotify_tracks_to_library(
        spotify_tracks,
        library,
        log,
    )

    stats = {
        "normalized_url": normalized_url,
        "total_tracks": len(spotify_tracks),
        "matched_tracks": len(matched_tracks),
        "unmatched_tracks": unmatched_count,
    }

    return matched_tracks, stats


def _collect_normalized_candidates(values):
    candidates = set()
    for raw_value in values:
        if not raw_value:
            continue
        normalized = _normalize_compare_value(raw_value)
        if normalized:
            candidates.add(normalized)
        stripped = _normalize_compare_value(_strip_parenthetical(raw_value))
        if stripped:
            candidates.add(stripped)
    return candidates


def _extract_year_token(value):
    if not value:
        return None
    match = re.search(r"(\d{4})", str(value))
    return match.group(1) if match else None

def _build_track_identity_key(track):
    raw_title = getattr(track, "title", None)
    raw_artist = getattr(track, "grandparentTitle", None)

    normalized_title = _normalize_compare_value(_strip_parenthetical(raw_title))
    if not normalized_title:
        normalized_title = _normalize_compare_value(raw_title)

    normalized_artist = _normalize_compare_value(raw_artist)

    if normalized_title or normalized_artist:
        return (normalized_title, normalized_artist)

    rating_key = getattr(track, "ratingKey", None)
    if rating_key is not None:
        return ("__rating_key__", str(rating_key))

    guid = getattr(track, "guid", None)
    if guid:
        return ("__guid__", str(guid))

    return ("__object_id__", str(id(track)))


def _coerce_non_negative_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric) or numeric < 0:
        return None

    return numeric


def _extract_numeric_from_candidates(container, field_names):
    if not isinstance(container, dict):
        return None

    for field in field_names:
        if field not in container:
            continue
        numeric = _coerce_non_negative_float(container.get(field))
        if numeric is not None:
            return numeric

    return None


def _build_album_identity_key(track):
    for attr_name in ("parentRatingKey", "parentGuid", "parentKey"):
        value = getattr(track, attr_name, None)
        if value:
            return ("id", str(value))

    artist = getattr(track, "grandparentTitle", None) or getattr(track, "originalTitle", None)
    album = getattr(track, "parentTitle", None)
    year = getattr(track, "parentYear", None) or getattr(track, "year", None)
    release_date = getattr(track, "originallyAvailableAt", None)

    return (
        "meta",
        str(artist or ""),
        str(album or ""),
        str(year or ""),
        str(release_date or ""),
    )


def _resolve_track_popularity_value(track, playlist_logger=None):
    """Return the Plex rating count for a track, if available."""

    fallback = None
    for attr_name in ("ratingCount", "parentRatingCount"):
        raw_value = getattr(track, attr_name, None)
        fallback = _coerce_non_negative_float(raw_value)
        if fallback is not None:
            break

    if fallback is not None and playlist_logger and hasattr(playlist_logger, "debug"):
        try:
            track_title = getattr(track, "title", "<unknown>")
            album_title = getattr(track, "parentTitle", "<unknown>")
            playlist_logger.debug(
                "Using Plex ratingCount (%s) for track '%s' (album='%s')",
                fallback,
                track_title,
                album_title,
            )
        except Exception:  # pragma: no cover - defensive logging guard
            pass

    return fallback


def _deduplicate_tracks(tracks, log):
    if not tracks:
        return tracks, {}, 0

    dedup_map = {}
    order = []
    popularity_cache = {}
    duplicates_removed = 0

    for idx, track in enumerate(tracks):
        key = _build_track_identity_key(track)

        current_entry = dedup_map.get(key)

        popularity = _resolve_track_popularity_value(
            track,
            playlist_logger=log,
        )

        rating_key = getattr(track, "ratingKey", None)
        if rating_key is not None and popularity is not None:
            popularity_cache[str(rating_key)] = popularity

        if current_entry is None:
            dedup_map[key] = {
                "track": track,
                "popularity": popularity,
                "index": idx,
            }
            order.append(key)
            continue

        existing_popularity = current_entry["popularity"]

        better_candidate = False
        if popularity is None and existing_popularity is None:
            better_candidate = False
        elif popularity is None:
            better_candidate = False
        elif existing_popularity is None:
            better_candidate = True
        else:
            better_candidate = popularity > existing_popularity

        if better_candidate:
            duplicates_removed += 1
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "Deduplicated track '%s' by '%s' – replacing with version from album '%s' (pop %.2f → %.2f)",
                    getattr(current_entry["track"], "title", "<unknown>"),
                    getattr(current_entry["track"], "grandparentTitle", "<unknown>"),
                    getattr(track, "parentTitle", getattr(current_entry["track"], "parentTitle", "<unknown album>")),
                    existing_popularity if existing_popularity is not None else float("nan"),
                    popularity,
                )
            dedup_map[key]["track"] = track
            dedup_map[key]["popularity"] = popularity
        else:
            duplicates_removed += 1
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "Deduplicated track '%s' by '%s' – keeping existing version from album '%s' (pop %.2f ≥ %.2f)",
                    getattr(track, "title", "<unknown>"),
                    getattr(track, "grandparentTitle", "<unknown>"),
                    getattr(current_entry["track"], "parentTitle", "<unknown album>"),
                    existing_popularity if existing_popularity is not None else float("nan"),
                    popularity if popularity is not None else float("nan"),
                )

    deduped_tracks = [dedup_map[key]["track"] for key in sorted(order, key=lambda item: dedup_map[item]["index"])]
    return deduped_tracks, popularity_cache, duplicates_removed


def _compute_album_popularity_boosts(
    tracks,
    popularity_cache,
    playlist_logger=None,
    top_5_boost=1.0,
):
    if not tracks:
        return {}, {}

    try:
        boost_value = float(top_5_boost)
    except (TypeError, ValueError):
        boost_value = 1.0

    if not math.isfinite(boost_value) or boost_value < 0:
        boost_value = 1.0

    album_map = defaultdict(list)
    adjusted_by_rating_key = {}
    adjusted_by_object = {}

    for track in tracks:
        cache_key = getattr(track, "ratingKey", None)
        cache_key_str = str(cache_key) if cache_key is not None else None
        popularity = None

        if cache_key_str and cache_key_str in popularity_cache:
            popularity = popularity_cache[cache_key_str]
        else:
            popularity = _resolve_track_popularity_value(
                track,
                playlist_logger=playlist_logger,
            )
            if cache_key_str:
                popularity_cache[cache_key_str] = popularity

        if popularity is None:
            continue

        album_key = _build_album_identity_key(track)
        album_map[album_key].append((track, popularity, cache_key_str))

    for album_tracks in album_map.values():
        album_tracks.sort(key=lambda entry: entry[1], reverse=True)
        for index, (track, base_score, cache_key_str) in enumerate(album_tracks):
            adjusted_score = base_score * boost_value if index < 5 else base_score
            if cache_key_str:
                adjusted_by_rating_key[cache_key_str] = adjusted_score
            else:
                adjusted_by_object[id(track)] = adjusted_score

    return adjusted_by_rating_key, adjusted_by_object


def _normalize_boost_condition(raw_condition):
    if not isinstance(raw_condition, dict):
        raw_condition = {}

    field = str(raw_condition.get("field") or "").strip()
    operator = str(raw_condition.get("operator") or "equals").strip() or "equals"
    value = raw_condition.get("value", "")

    match_all_raw = raw_condition.get("match_all")
    if isinstance(match_all_raw, bool):
        match_all_value = match_all_raw
    elif match_all_raw is None:
        match_all_value = True
    else:
        match_all_value = bool(match_all_raw)

    has_multiple_expected_values = False
    if isinstance(value, str):
        segments = [segment.strip() for segment in value.split(",")]
        has_multiple_expected_values = len([segment for segment in segments if segment]) > 1
    elif isinstance(value, (list, tuple, set)):
        has_multiple_expected_values = len(value) > 1

    if has_multiple_expected_values:
        match_all_value = False

    return {
        "field": field,
        "operator": operator,
        "value": value,
        "match_all": match_all_value,
    }


def _normalize_sort_option(raw_value):
    if not isinstance(raw_value, str):
        return raw_value

    trimmed = raw_value.strip()
    if not trimmed:
        return None

    normalized = trimmed.lower()
    known_options = {
        "popularity": "popularity",
        "alphabetical": "alphabetical",
        "reverse_alphabetical": "reverse_alphabetical",
        "oldest_first": "oldest_first",
        "newest_first": "newest_first",
    }

    return known_options.get(normalized, trimmed)


def _apply_configured_popularity_boosts(
    tracks,
    boost_rules,
    dedup_popularity_cache,
    album_popularity_cache,
    album_popularity_cache_by_object,
    playlist_logger=None,
):
    if not tracks or not boost_rules:
        return {}, {}

    normalized_rules = []
    for rule in boost_rules:
        if not isinstance(rule, dict):
            continue

        raw_conditions = rule.get("conditions")
        if isinstance(raw_conditions, (list, tuple)):
            normalized_conditions = [
                _normalize_boost_condition(condition)
                for condition in raw_conditions
            ]
        else:
            normalized_conditions = [_normalize_boost_condition(rule)]

        normalized_conditions = [
            condition for condition in normalized_conditions if condition["field"]
        ]

        compiled_conditions = [
            _compile_filter_entry(condition) for condition in normalized_conditions
        ]

        if not compiled_conditions:
            continue

        boost_value = _coerce_non_negative_float(rule.get("boost"))
        if boost_value is None:
            boost_value = 1.0
        normalized_rules.append(
            {
                "conditions": compiled_conditions,
                "boost": boost_value,
            }
        )

    if not normalized_rules:
        return {}, {}

    adjustments_by_key = {}
    adjustments_by_object = {}

    for track in tracks:
        cache_key = getattr(track, "ratingKey", None)
        cache_key_str = str(cache_key) if cache_key is not None else None
        object_key = id(track)

        if cache_key_str and cache_key_str in album_popularity_cache:
            base_score = album_popularity_cache[cache_key_str]
        elif cache_key_str and cache_key_str in dedup_popularity_cache:
            base_score = dedup_popularity_cache[cache_key_str]
        elif object_key in album_popularity_cache_by_object:
            base_score = album_popularity_cache_by_object[object_key]
        else:
            base_score = _resolve_track_popularity_value(
                track,
                playlist_logger=playlist_logger,
            )
            if cache_key_str is not None and base_score is not None:
                dedup_popularity_cache[cache_key_str] = base_score

        if base_score is None:
            continue

        try:
            base_numeric = float(base_score)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(base_numeric):
            continue

        multiplier = 1.0
        for rule in normalized_rules:
            if all(
                check_condition(
                    get_field_value(track, condition.field),
                    condition.operator,
                    condition.expected,
                    condition.match_all,
                    compiled=condition,
                )
                for condition in rule["conditions"]
            ):
                multiplier *= rule["boost"]

        if multiplier == 1.0:
            continue

        adjusted_score = base_numeric * multiplier

        if cache_key_str is not None:
            dedup_popularity_cache[cache_key_str] = adjusted_score
            adjustments_by_key[cache_key_str] = adjusted_score
            if cache_key_str in album_popularity_cache:
                album_popularity_cache[cache_key_str] = adjusted_score
        else:
            adjustments_by_object[object_key] = adjusted_score
            album_popularity_cache_by_object[object_key] = adjusted_score

    return adjustments_by_key, adjustments_by_object


def _resolve_popularity_for_sort(
    track,
    cache_key_str,
    object_cache_key,
    dedup_popularity_cache,
    album_popularity_cache,
    album_popularity_cache_by_object,
    playlist_logger=None,
    sort_desc=True,
):
    """Return the popularity score used for sorting and update caches."""

    popularity_value = None

    if cache_key_str and cache_key_str in album_popularity_cache:
        popularity_value = album_popularity_cache[cache_key_str]
    elif cache_key_str is None and object_cache_key in album_popularity_cache_by_object:
        popularity_value = album_popularity_cache_by_object[object_cache_key]
    elif cache_key_str and cache_key_str in dedup_popularity_cache:
        popularity_value = dedup_popularity_cache[cache_key_str]
    else:
        popularity_value = _resolve_track_popularity_value(
            track,
            playlist_logger=playlist_logger,
        )

    if popularity_value is None:
        sentinel = float("-inf") if sort_desc else float("inf")
        return sentinel, False

    if cache_key_str is not None:
        dedup_popularity_cache[cache_key_str] = popularity_value
        album_popularity_cache[cache_key_str] = popularity_value
    else:
        album_popularity_cache_by_object[object_cache_key] = popularity_value

    return popularity_value, True


def _normalize_release_date(value):
    if not value:
        return None

    if isinstance(value, datetime.datetime):
        return value.date()

    if isinstance(value, datetime.date):
        return value

    text = str(value)

    match = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", text)
    if match:
        try:
            return datetime.date(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
        except ValueError:
            pass

    match = re.search(r"(\d{4})[-/](\d{2})", text)
    if match:
        try:
            return datetime.date(int(match.group(1)), int(match.group(2)), 1)
        except ValueError:
            pass

    match = re.search(r"(\d{4})", text)
    if match:
        try:
            return datetime.date(int(match.group(1)), 1, 1)
        except ValueError:
            pass

    return None


def _resolve_album_release_date(track):
    """Return the best-known release date for the provided track's album."""

    date_candidates = [
        getattr(track, "parentOriginallyAvailableAt", None),
        getattr(track, "originallyAvailableAt", None),
    ]

    for candidate in date_candidates:
        normalized = _normalize_release_date(candidate)
        if normalized:
            return normalized

    return None


def _resolve_album_year(track):
    """Return the best-known release year for the provided track's album."""

    def _normalize_year(value):
        token = _extract_year_token(value)
        return token

    cache_keys = []
    parent_rating_key = getattr(track, "parentRatingKey", None)
    if parent_rating_key is not None:
        cache_keys.append(f"album:{parent_rating_key}")
    track_rating_key = getattr(track, "ratingKey", None)
    if track_rating_key is not None:
        cache_keys.append(f"track:{track_rating_key}")

    for key in cache_keys:
        cached_year = _lookup_cached_value(_ALBUM_YEAR_CACHE, _ALBUM_YEAR_MISS_KEYS, key)
        if cached_year is not _CACHE_LOOKUP_SENTINEL:
            return cached_year

    year_candidates = [
        getattr(track, "parentYear", None),
        getattr(track, "year", None),
        getattr(track, "originallyAvailableAt", None),
        getattr(track, "parentOriginallyAvailableAt", None),
    ]

    for candidate in year_candidates:
        normalized = _normalize_year(candidate)
        if normalized:
            for key in cache_keys:
                _store_cached_value(_ALBUM_YEAR_CACHE, _ALBUM_YEAR_MISS_KEYS, key, normalized)
            return normalized

    if CACHE_ONLY:
        for key in cache_keys:
            _store_cached_value(_ALBUM_YEAR_CACHE, _ALBUM_YEAR_MISS_KEYS, key, None)
        return None

    log = _get_active_logger()

    rating_key = getattr(track, "ratingKey", None)
    if rating_key:
        try:
            track_xml = fetch_full_metadata(rating_key)
        except Exception as exc:  # pragma: no cover - network/cache errors
            log.debug(
                "Unable to fetch track metadata for ratingKey=%s while resolving album year: %s",
                rating_key,
                exc,
            )
        else:
            for field in (
                "parentYear",
                "year",
                "originallyAvailableAt",
                "parentOriginallyAvailableAt",
            ):
                normalized = _normalize_year(parse_field_from_xml(track_xml, field))
                if normalized:
                    for key in cache_keys:
                        _store_cached_value(_ALBUM_YEAR_CACHE, _ALBUM_YEAR_MISS_KEYS, key, normalized)
                    return normalized

    parent_rating_key = getattr(track, "parentRatingKey", None)
    if parent_rating_key:
        try:
            album_xml = fetch_full_metadata(parent_rating_key)
        except Exception as exc:  # pragma: no cover - network/cache errors
            log.debug(
                "Unable to fetch album metadata for ratingKey=%s while resolving album year: %s",
                parent_rating_key,
                exc,
            )
        else:
            for field in ("year", "parentYear", "originallyAvailableAt"):
                normalized = _normalize_year(parse_field_from_xml(album_xml, field))
                if normalized:
                    for key in cache_keys:
                        _store_cached_value(_ALBUM_YEAR_CACHE, _ALBUM_YEAR_MISS_KEYS, key, normalized)
                    return normalized

    for key in cache_keys:
        _store_cached_value(_ALBUM_YEAR_CACHE, _ALBUM_YEAR_MISS_KEYS, key, None)
    return None


def _get_album_guid(track):
    """Return the album GUID (``plex://album/...``) for a track, if available."""

    cache_keys = []
    parent_rating_key = getattr(track, "parentRatingKey", None)
    if parent_rating_key is not None:
        cache_keys.append(f"album:{parent_rating_key}")
    track_rating_key = getattr(track, "ratingKey", None)
    if track_rating_key is not None:
        cache_keys.append(f"track:{track_rating_key}")

    for key in cache_keys:
        cached_guid = _lookup_cached_value(_ALBUM_GUID_CACHE, _ALBUM_GUID_MISS_KEYS, key)
        if cached_guid is not _CACHE_LOOKUP_SENTINEL:
            return cached_guid

    parent_guid = getattr(track, "parentGuid", None)
    if parent_guid:
        guid_value = str(parent_guid)
        for key in cache_keys:
            _store_cached_value(_ALBUM_GUID_CACHE, _ALBUM_GUID_MISS_KEYS, key, guid_value)
        return guid_value

    # Fall back to the track metadata and, if necessary, the album metadata.
    try:
        track_xml = fetch_full_metadata(track.ratingKey)
        parent_guid = parse_field_from_xml(track_xml, "parentGuid")
        if parent_guid:
            guid_value = str(parent_guid)
            for key in cache_keys:
                _store_cached_value(_ALBUM_GUID_CACHE, _ALBUM_GUID_MISS_KEYS, key, guid_value)
            return guid_value
    except Exception as exc:
        logger.debug(
            "Unable to resolve parentGuid from track metadata for ratingKey=%s: %s",
            getattr(track, "ratingKey", "<no-key>"),
            exc,
        )

    parent_rating_key = getattr(track, "parentRatingKey", None)
    if not parent_rating_key:
        return None

    try:
        album_xml = fetch_full_metadata(parent_rating_key)
    except Exception as exc:
        logger.debug(
            "Unable to fetch album metadata for ratingKey=%s: %s",
            parent_rating_key,
            exc,
        )
        return None

    for candidate in ("guid", "parentGuid"):
        candidate_guid = parse_field_from_xml(album_xml, candidate)
        if candidate_guid:
            guid_value = str(candidate_guid)
            for key in cache_keys:
                _store_cached_value(_ALBUM_GUID_CACHE, _ALBUM_GUID_MISS_KEYS, key, guid_value)
            return guid_value

    for key in cache_keys:
        _store_cached_value(_ALBUM_GUID_CACHE, _ALBUM_GUID_MISS_KEYS, key, None)
    return None


def _get_track_guid(track):
    """Return the track GUID (``plex://track/...``) for a track, if available."""

    cache_key = None
    rating_key = getattr(track, "ratingKey", None)
    if rating_key is not None:
        cache_key = f"track:{rating_key}"

    cached_guid = _lookup_cached_value(_TRACK_GUID_CACHE, _TRACK_GUID_MISS_KEYS, cache_key)
    if cached_guid is not _CACHE_LOOKUP_SENTINEL:
        return cached_guid

    track_guid = getattr(track, "guid", None)
    if track_guid:
        guid_value = str(track_guid)
        _store_cached_value(_TRACK_GUID_CACHE, _TRACK_GUID_MISS_KEYS, cache_key, guid_value)
        return guid_value

    try:
        track_xml = fetch_full_metadata(track.ratingKey)
    except Exception as exc:
        logger.debug(
            "Unable to fetch track metadata for ratingKey=%s: %s",
            getattr(track, "ratingKey", "<no-key>"),
            exc,
        )
        return None

    guid = parse_field_from_xml(track_xml, "guid")
    guid_value = str(guid) if guid else None
    _store_cached_value(_TRACK_GUID_CACHE, _TRACK_GUID_MISS_KEYS, cache_key, guid_value)
    return guid_value


def _extract_guid_from_search_results(xml_text, album_norms, artist_norms, target_year):
    log = _get_active_logger()
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        log.debug("Canonical search XML parse error.")
        return None

    candidate_nodes = []
    for tag in ("Directory", "Metadata"):
        candidate_nodes.extend(root.findall(f".//{tag}"))

    for node in candidate_nodes:
        node_type = node.attrib.get("type")
        if node_type and node_type not in {"album", "artist", "collection"}:
            if node_type != "album":
                continue

        guid = node.attrib.get("guid") or node.attrib.get("parentGuid")
        if not guid:
            continue

        candidate_album_norms = _collect_normalized_candidates(
            [
                node.attrib.get("title"),
                node.attrib.get("parentTitle"),
                node.attrib.get("originalTitle"),
                node.attrib.get("albumTitle"),
            ]
        )
        if album_norms and not (album_norms & candidate_album_norms):
            continue

        if artist_norms:
            candidate_artist_norms = _collect_normalized_candidates(
                [
                    node.attrib.get("grandparentTitle"),
                    node.attrib.get("parentTitle"),
                    node.attrib.get("artistTitle"),
                    node.attrib.get("primaryArtist"),
                ]
            )
            if candidate_artist_norms and not (artist_norms & candidate_artist_norms):
                continue
            if not candidate_artist_norms:
                continue

        if target_year:
            candidate_year_tokens = set()
            for attr_name in ("year", "parentYear", "originallyAvailableAt"):
                year_token = _extract_year_token(node.attrib.get(attr_name))
                if year_token:
                    candidate_year_tokens.add(year_token)
            if candidate_year_tokens and target_year not in candidate_year_tokens:
                continue

        return guid

    return None


class AllMusicPopularityProvider:
    """Resolve track popularity using AllMusic search results."""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._enabled = bool(ALLMUSIC_ENABLED)
        self._error = None
        self._session = None
        self._timeout = ALLMUSIC_TIMEOUT or 10
        self._cache_file = ALLMUSIC_CACHE_FILE
        self._track_cache = {}
        self._query_cache = {}
        self._album_cache = {}
        self._album_track_scores = {}
        self._album_track_details = {}
        self._album_single_sets = {}
        self._cache_lock = threading.Lock()
        self._album_scores_lock = threading.Lock()
        self._cache_dirty = False
        self._google_min_interval = max(0.0, ALLMUSIC_GOOGLE_MIN_INTERVAL)
        self._google_backoff = max(0.0, ALLMUSIC_GOOGLE_BACKOFF)
        self._google_rate_lock = threading.Lock()
        self._google_next_allowed = 0.0

        if not self._enabled:
            self._error = "AllMusic integration disabled via configuration."
            return

        try:
            self._session = requests.Session()
            if ALLMUSIC_USER_AGENT:
                self._session.headers.update({"User-Agent": ALLMUSIC_USER_AGENT})
        except Exception as exc:  # pragma: no cover - defensive only
            self._error = f"Failed to initialize HTTP session: {exc}"
            self._enabled = False
            return

        self._load_cache()

    @classmethod
    def get_shared(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @classmethod
    def save_shared_cache(cls):
        instance = cls._instance
        if instance is not None:
            instance.save_cache()

    @property
    def is_enabled(self):
        return bool(self._enabled and self._session)

    def describe_error(self):
        return self._error

    def save_cache(self):
        if not self._cache_file or not self._cache_dirty:
            return

        cache_dir = os.path.dirname(self._cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        payload = {
            "version": ALLMUSIC_CACHE_VERSION,
            "tracks": self._track_cache,
            "queries": self._query_cache,
            "albums": self._album_cache,
        }

        try:
            with open(self._cache_file, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - IO safety
            logger.warning("Unable to persist AllMusic cache to '%s': %s", self._cache_file, exc)
        else:
            self._cache_dirty = False

    def _load_cache(self):
        if not self._cache_file or not os.path.exists(self._cache_file):
            return

        try:
            with open(self._cache_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            logger.warning("Unable to load AllMusic cache from '%s': %s", self._cache_file, exc)
            return

        if not isinstance(payload, dict):
            return

        if payload.get("version") != ALLMUSIC_CACHE_VERSION:
            logger.debug("Ignoring AllMusic cache with mismatched version (%s)", payload.get("version"))
            return

        self._track_cache = payload.get("tracks", {}) or {}
        self._query_cache = payload.get("queries", {}) or {}
        self._album_cache = payload.get("albums", {}) or {}

    def get_popularity(self, track, playlist_logger=None):
        title = getattr(track, "title", None)
        artist = getattr(track, "grandparentTitle", None)
        album = getattr(track, "parentTitle", None)

        track_key = self._build_track_key(title, artist, album)

        base_score = _resolve_track_popularity_value(
            track,
            playlist_logger=playlist_logger,
        )

        components = {
            "rating_count": base_score,
            "base_score": base_score,
            "fallback": None if base_score is not None else "no_data",
        }

        details = {
            "title": title,
            "artist": artist,
            "album": album,
            "rating_count": base_score,
            "components": components,
        }

        final_score, finalized_details = self._apply_single_boost(
            track,
            track_key,
            base_score,
            details,
        )

        self._remember_track_cache(track_key, final_score, finalized_details)
        return final_score

    def _remember_track_cache(self, track_key, popularity, details):
        if not track_key:
            return
        with self._cache_lock:
            self._track_cache[track_key] = {
                "popularity": popularity,
                "timestamp": time.time(),
                "details": details or {},
            }
            self._cache_dirty = True

    def _build_track_key(self, title, artist, album=None):
        if not title or not artist:
            return None

        base_key = f"{_normalize_compare_value(title)}::{_normalize_compare_value(artist)}"
        album_norm = _normalize_compare_value(_strip_parenthetical(album)) if album else ""

        if album_norm:
            return f"{base_key}::{album_norm}"
        return base_key

    def _build_query(self, title, artist, album):
        parts = []

        if album:
            cleaned_album = _strip_parenthetical(album)
            if cleaned_album:
                parts.append(cleaned_album.strip())

        if artist:
            cleaned_artist = str(artist).strip()
            if cleaned_artist:
                parts.append(cleaned_artist)

        if not parts and title:
            cleaned_title = str(title).strip()
            if cleaned_title:
                parts.append(cleaned_title)

        if not parts:
            return None

        return " ".join(parts)

    def _execute_search(self, query, title, artist, album, playlist_logger=None):
        if not self._session or not album:
            return None

        last_error = None
        final_details = None
        used_query = None

        album_info = {
            "album": _strip_parenthetical(album) if album else album,
            "artist": artist,
        }

        for variant in self._iter_query_variants(query):
            google_query = self._build_google_search_query(variant)
            if not google_query:
                continue

            try:
                response_html = self._perform_google_search(
                    google_query,
                    original_query=variant,
                    playlist_logger=playlist_logger,
                )
            except Exception as exc:
                last_error = exc
                self._error = str(exc)
                if playlist_logger:
                    playlist_logger.debug(
                        "Google search for '%s' failed: %s",
                        google_query,
                        exc,
                    )
                else:
                    logger.debug("Google search for '%s' failed: %s", google_query, exc)
                self._remember_query_cache_entry(variant, None)
                continue

            album_url = self._extract_first_allmusic_album_url(response_html)

            if not album_url:
                self._remember_query_cache_entry(variant, None)
                continue

            if playlist_logger:
                playlist_logger.debug("AllMusic album page URL: %s", album_url)

            metadata = self._get_album_metadata(
                album_url,
                playlist_logger=playlist_logger,
            )

            rating_count = metadata.get("rating_count")

            artists = []
            if artist:
                artists.append(str(artist).strip())

            cleaned_album_title = _strip_parenthetical(album) if album else None

            details = {
                "title": title,
                "artists": artists,
                "album": cleaned_album_title or None,
                "album_url": album_url,
                "rating_count": rating_count,
                "album_rating_count": rating_count,
                "source": "google_album_search",
            }

            self._remember_query_cache_entry(variant, details)

            used_query = variant
            final_details = details
            break

        if final_details is None:
            if last_error is not None:
                self._error = str(last_error)
            self._remember_query_cache_entry(query, None)
            return None

        if used_query != query:
            self._remember_query_cache_entry(query, final_details)

        return final_details

    def _remember_query_cache_entry(self, query, details):
        with self._cache_lock:
            self._query_cache[query] = details or {}
            self._cache_dirty = True

    def _resolve_via_album_page(self, album, artist, track_title, playlist_logger=None):
        album_query_parts = []
        if album:
            album_query_parts.append(_strip_parenthetical(album))
        if artist:
            album_query_parts.append(str(artist).strip())

        album_query_parts = [part for part in album_query_parts if part]
        if not album_query_parts:
            return None

        album_query = " ".join(album_query_parts)
        last_error = None

        for variant in self._iter_query_variants(album_query):
            google_query = self._build_google_search_query(variant)
            if not google_query:
                continue

            try:
                response_html = self._perform_google_search(
                    google_query,
                    original_query=variant,
                    playlist_logger=playlist_logger,
                )
            except Exception as exc:
                last_error = exc
                if playlist_logger:
                    playlist_logger.debug(
                        "Google search for '%s' failed: %s",
                        google_query,
                        exc,
                    )
                else:
                    logger.debug("Google search for '%s' failed: %s", google_query, exc)
                continue

            album_url = self._extract_first_allmusic_album_url(response_html)

            if not album_url:
                continue

            metadata = self._get_album_metadata(
                album_url,
                playlist_logger=playlist_logger,
            )

            artists = []
            if artist:
                artists.append(str(artist).strip())

            details = {
                "title": track_title,
                "artists": artists,
                "album": _strip_parenthetical(album) if album else album,
                "album_url": album_url,
                "rating_count": metadata.get("rating_count"),
                "album_rating_count": metadata.get("rating_count"),
                "source": "google_album_search",
            }

            if playlist_logger:
                playlist_logger.debug("AllMusic album page URL: %s", album_url)
                playlist_logger.debug(
                    "AllMusic album fallback matched '%s' → album_url='%s'",
                    details.get("album"),
                    album_url,
                )
            else:
                logger.debug("AllMusic album page URL: %s", album_url)
                logger.debug(
                    "AllMusic album fallback matched '%s' → album_url='%s'",
                    details.get("album"),
                    album_url,
                )

            return details

        if last_error is not None:
            self._error = str(last_error)

        return None

    def _iter_query_variants(self, query):
        seen = set()

        def _normalize(value):
            if not value:
                return None
            collapsed = " ".join(value.split())
            return collapsed or None

        def _add(value):
            normalized = _normalize(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                variants.append(normalized)

        variants = []
        _add(query)

        if query:
            without_parentheses = re.sub(r"\s*\([^)]*\)", "", query)
            _add(without_parentheses)

            ascii_query = unicodedata.normalize("NFKD", query).encode("ascii", "ignore").decode(
                "ascii"
            )
            _add(ascii_query)

            ascii_without_parentheses = re.sub(r"\s*\([^)]*\)", "", ascii_query)
            _add(ascii_without_parentheses)

        for variant in variants:
            yield variant

    @staticmethod
    def _build_google_search_query(value):
        if not value:
            return None

        without_parentheses = re.sub(r"\s*\([^)]*\)", " ", value)
        cleaned = " ".join(without_parentheses.split())
        if not cleaned:
            return None

        return f"allmusic.com: {cleaned}"

    def _perform_google_search(self, query, original_query=None, playlist_logger=None):
        logger_fn = playlist_logger.debug if playlist_logger else logger.debug
        logger_fn("Google search query: %s", query)

        self._throttle_google_request()

        try:
            response = self._session.get(
                "https://www.google.com/search",
                params={"q": query, "num": "5", "hl": "en"},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code == 429:
                self._register_google_backoff()
                fallback_source = original_query or query
                logger_fn(
                    "Google search returned HTTP 429; attempting direct AllMusic search for '%s'",
                    fallback_source,
                )
                fallback_html = self._perform_allmusic_search(
                    fallback_source, playlist_logger=playlist_logger
                )
                if fallback_html:
                    return fallback_html
            raise

        return response.text

    def _throttle_google_request(self):
        min_interval = self._google_min_interval
        if min_interval <= 0:
            return

        while True:
            now = time.monotonic()
            with self._google_rate_lock:
                wait_time = self._google_next_allowed - now
                if wait_time <= 0:
                    self._google_next_allowed = now + min_interval
                    return

            sleep_for = min(wait_time, min_interval)
            if sleep_for > 0:
                logger.debug(
                    "Throttling Google search request for %.2fs to respect rate limits",
                    sleep_for,
                )
                time.sleep(sleep_for)

    def _register_google_backoff(self):
        backoff = self._google_backoff
        if backoff <= 0:
            return

        now = time.monotonic()
        with self._google_rate_lock:
            next_allowed = now + backoff
            if next_allowed > self._google_next_allowed:
                self._google_next_allowed = next_allowed

    def _perform_allmusic_search(self, value, playlist_logger=None):
        if not value:
            return None

        search_term = value
        if value.lower().startswith("allmusic.com:"):
            search_term = value.split(":", 1)[1].strip()

        if not search_term:
            return None

        logger_fn = playlist_logger.debug if playlist_logger else logger.debug
        logger_fn("AllMusic fallback search query: %s", search_term)

        encoded_term = quote(search_term, safe="")
        response = self._session.get(
            f"https://www.allmusic.com/search/all/{encoded_term}",
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.text

    def _extract_first_allmusic_album_url(self, html):
        if not html:
            return None

        patterns = [
            r'href="/url\?q=(https?://www\.allmusic\.com/(?:album|release)/[^"&]+)',
            r'href="(https?://www\.allmusic\.com/(?:album|release)/[^"&]+)"',
        ]

        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if not match:
                continue
            raw_url = unquote(match.group(1))
            normalized = self._normalize_album_url(raw_url)
            if normalized:
                return normalized

        return None

    def _select_candidate(self, html, track_info, playlist_logger=None):
        candidates = list(self._iter_candidates(html))
        if not candidates:
            return None

        track_title_norm = _normalize_compare_value(track_info.get("title"))
        track_artist_norms = _collect_normalized_candidates([track_info.get("artist")])
        track_album_norms = _collect_normalized_candidates([track_info.get("album")])

        best_candidate = None
        best_score = float("-inf")

        for candidate in candidates:
            candidate_title_norm = _normalize_compare_value(candidate.get("title"))
            candidate_artist_norms = _collect_normalized_candidates(candidate.get("artists", []))
            candidate_album_norms = _collect_normalized_candidates(candidate.get("albums", []))

            popularity = candidate.get("popularity") or 0
            try:
                popularity_score = float(popularity) / 100.0
            except (TypeError, ValueError):
                popularity_score = 0.0

            score = popularity_score

            if track_title_norm and candidate_title_norm == track_title_norm:
                score += 5
            elif track_title_norm and candidate_title_norm and track_title_norm in candidate_title_norm:
                score += 2

            if track_artist_norms and candidate_artist_norms & track_artist_norms:
                score += 3

            if track_album_norms and candidate_album_norms & track_album_norms:
                score += 1

            candidate["_score"] = score

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if not best_candidate:
            return None

        rating_details = best_candidate.get("_rating_details") or {}
        album_url = best_candidate.get("album_url")
        album_metadata = {}

        if album_url:
            album_metadata = self._get_album_metadata(
                album_url,
                playlist_logger=playlist_logger,
            )

        details = {
            "title": best_candidate.get("title"),
            "artists": list(best_candidate.get("artists", [])),
            "album": best_candidate.get("albums", [None])[0] if best_candidate.get("albums") else None,
            "score": best_candidate.get("_score"),
            "rating_count": rating_details.get("rating_count"),
            "rating": rating_details.get("rating"),
            "rating_scale": rating_details.get("rating_scale"),
            "album_url": album_url,
            "album_rating_count": album_metadata.get("rating_count"),
        }

        return details

    def _iter_candidates(self, html):
        if not html:
            return

        row_pattern = re.compile(
            r"<tr[^>]*class=\"[^\"]*(?:song|track)[^\"]*\"[^>]*>(.*?)</tr>",
            re.IGNORECASE | re.DOTALL,
        )

        for block in row_pattern.findall(html):
            title = self._extract_text(block, r'class=\"title\"[^>]*>(.*?)</a>')
            if not title:
                title = self._extract_text(block, r'class=\"title\"[^>]*>(.*?)</td>')

            artist_text = self._extract_text(block, r'class=\"performers\"[^>]*>(.*?)</td>')
            if not artist_text:
                artist_text = self._extract_text(block, r'class=\"artist\"[^>]*>(.*?)</td>')

            album_text = self._extract_text(block, r'class=\"release\"[^>]*>(.*?)</td>')
            album_url = self._extract_album_url(block)

            rating_details = self._extract_rating_details(block)

            artists = self._split_artists(artist_text)
            albums = [album_text] if album_text else []

            yield {
                "title": title,
                "artists": artists,
                "albums": albums,
                "_rating_details": rating_details,
                "album_url": album_url,
            }

    def _iter_album_search_candidates(self, html):
        if not html:
            return

        row_pattern = re.compile(
            r"<tr[^>]*class=\"[^\"]*(?:album|release)[^\"]*\"[^>]*>(.*?)</tr>",
            re.IGNORECASE | re.DOTALL,
        )

        for block in row_pattern.findall(html):
            album_title = self._extract_text(block, r'class=\"title\"[^>]*>(.*?)</a>')
            if not album_title:
                album_title = self._extract_text(block, r'class=\"title\"[^>]*>(.*?)</td>')

            artist_text = self._extract_text(block, r'class=\"artist\"[^>]*>(.*?)</td>')
            if not artist_text:
                artist_text = self._extract_text(block, r'class=\"performers\"[^>]*>(.*?)</td>')

            album_url = self._extract_album_result_url(block) or self._extract_album_url(block)

            artists = self._split_artists(artist_text)
            cleaned_album = _strip_parenthetical(album_title) if album_title else None

            if album_title or album_url:
                yield {
                    "album": cleaned_album or album_title,
                    "raw_album": album_title,
                    "artists": artists,
                    "album_url": album_url,
                }

    def _select_album_candidate(self, html, album_info):
        candidates = list(self._iter_album_search_candidates(html))
        if not candidates:
            return None

        target_album_norm = _normalize_compare_value(
            _strip_parenthetical(album_info.get("album"))
        )
        target_artist_norms = _collect_normalized_candidates([album_info.get("artist")])

        best_candidate = None
        best_score = float("-inf")

        for candidate in candidates:
            candidate_album_norm = _normalize_compare_value(
                _strip_parenthetical(candidate.get("raw_album") or candidate.get("album"))
            )
            candidate_artist_norms = _collect_normalized_candidates(candidate.get("artists", []))

            score = 0.0

            if target_album_norm and candidate_album_norm == target_album_norm:
                score += 5
            elif target_album_norm and candidate_album_norm and target_album_norm in candidate_album_norm:
                score += 2

            if target_artist_norms and candidate_artist_norms & target_artist_norms:
                score += 3

            if candidate.get("album_url"):
                score += 1

            candidate["_score"] = score

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    @staticmethod
    def _extract_text(block, pattern):
        match = re.search(pattern, block, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        text = re.sub(r"<[^>]+>", " ", match.group(1))
        return unescape(text).strip()

    def _extract_album_url(self, block):
        match = re.search(
            r'class=\"release\"[^>]*>\s*<a[^>]*href=\"([^\"]+)\"',
            block,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        return self._normalize_album_url(match.group(1))

    def _extract_album_result_url(self, block):
        match = re.search(
            r'<a[^>]*href=\"([^\"]+/album/[^\"]+)\"',
            block,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        return self._normalize_album_url(match.group(1))

    @staticmethod
    def _normalize_album_url(url):
        if not url:
            return None
        cleaned = url.strip()
        if cleaned.startswith("//"):
            return f"https:{cleaned}"
        if cleaned.startswith("/"):
            return f"https://www.allmusic.com{cleaned}"
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            return cleaned
        return f"https://www.allmusic.com/{cleaned.lstrip('/')}"

    def _split_artists(self, text):
        if not text:
            return []
        parts = re.split(r"[,/;&]+", text)
        cleaned = [part.strip() for part in parts if part and part.strip()]
        return cleaned

    def _extract_rating_details(self, block):
        details = {
            "rating_count": None,
            "rating": None,
            "rating_scale": None,
        }

        match = re.search(r'data-ratingcount="([0-9,]+)"', block, re.IGNORECASE)
        if match:
            try:
                details["rating_count"] = int(match.group(1).replace(",", ""))
            except ValueError:
                details["rating_count"] = None

        # Direct numeric rating attribute
        match = re.search(r'data-(?:avg-)?rating="([0-9.]+)"', block, re.IGNORECASE)
        if match:
            details["rating_scale"] = "stars"
            details["rating"] = self._convert_rating_to_popularity(match.group(1), scale="stars")
            return details

        # CSS class based rating with half-star granularity (e.g. rating-45 → 4.5 stars)
        match = re.search(r'allmusic-rating[^>]*rating-([0-9]+)', block, re.IGNORECASE)
        if match:
            details["rating_scale"] = "half-stars"
            details["rating"] = self._convert_rating_to_popularity(match.group(1), scale="half-stars")
            return details

        match = re.search(r'rating-[^>]*?(\d+)', block, re.IGNORECASE)
        if match:
            details["rating_scale"] = "half-stars"
            details["rating"] = self._convert_rating_to_popularity(match.group(1), scale="half-stars")
            return details

        return details

    def _convert_rating_to_popularity(self, value, scale="stars"):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

        if scale == "percentage":
            popularity = numeric
        elif scale == "half-stars":
            stars = numeric / 10.0 if numeric > 5 else numeric
            popularity = (stars / 5.0) * 100.0
        else:  # default treat as star count (0-5)
            popularity = (numeric / 5.0) * 100.0

        popularity = max(0.0, min(100.0, popularity))
        return round(popularity, 2)

    def _get_album_metadata(self, album_url, playlist_logger=None):
        normalized_url = self._normalize_album_url(album_url)
        if not normalized_url:
            return {}

        with self._cache_lock:
            cached_value = self._album_cache.get(normalized_url)

        if isinstance(cached_value, dict):
            return dict(cached_value)
        if cached_value is not None:
            # Backward compatibility with previous cache versions storing raw integers
            return {"rating_count": cached_value}

        metadata = {}

        try:
            response = self._session.get(normalized_url, timeout=self._timeout)
            response.raise_for_status()
            metadata = self._parse_album_metadata(response.text)
        except Exception as exc:
            if playlist_logger:
                playlist_logger.debug(
                    "Failed to fetch AllMusic album page '%s': %s",
                    normalized_url,
                    exc,
                )
            else:
                logger.debug("Failed to fetch AllMusic album page '%s': %s", normalized_url, exc)
            metadata = {}

        with self._cache_lock:
            self._album_cache[normalized_url] = metadata
            self._cache_dirty = True

        return dict(metadata)

    @classmethod
    def _parse_album_metadata(cls, html):
        rating_count = cls._parse_album_rating_count(html)

        metadata = {}
        if rating_count is not None:
            metadata["rating_count"] = rating_count

        return metadata

    @staticmethod
    def _parse_album_rating_count(html):
        if not html:
            return None

        patterns = [
            r'data-ratingcount="([0-9,]+)"',
            r'class=\"rating-count\"[^>]*>([0-9,]+)<',
            r'User Ratings[^<]*<[^>]*class=\"count\"[^>]*>([0-9,]+)<',
        ]

        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1).replace(",", ""))
                except (ValueError, TypeError):
                    continue

        return None


    @staticmethod
    def _fallback_track_identifier(track):
        rating_key = getattr(track, "ratingKey", None)
        if rating_key is not None:
            return f"ratingKey::{rating_key}"
        guid = getattr(track, "guid", None)
        if guid:
            return f"guid::{guid}"
        return None

    def _build_album_key(self, artist, album):
        normalized_artist = _normalize_compare_value(artist)
        normalized_album = _normalize_compare_value(_strip_parenthetical(album))
        if not normalized_artist or not normalized_album:
            return None
        return f"{normalized_artist}::{normalized_album}"

    @staticmethod
    def _compute_album_single_set(album_scores):
        ranked = sorted(
            ((track_id, score) for track_id, score in album_scores.items() if score is not None),
            key=lambda item: item[1],
            reverse=True,
        )
        return {track_id for track_id, _ in ranked[:5]}

    def _apply_single_boost(self, track, track_key, base_score, details):
        updated_details = dict(details or {})
        components = dict(updated_details.get("components") or {})
        updated_details["components"] = components
        components["base_score"] = base_score

        album_title = getattr(track, "parentTitle", None)
        artist = getattr(track, "grandparentTitle", None)
        album_key = self._build_album_key(artist, album_title)
        track_identifier = track_key or self._fallback_track_identifier(track)

        is_single = False
        multiplier = 1.0
        final_score = base_score
        singles_changed = False

        if album_key and track_identifier:
            with self._album_scores_lock:
                album_scores = self._album_track_scores.setdefault(album_key, {})
                album_scores[track_identifier] = base_score

                album_details = self._album_track_details.setdefault(album_key, {})
                album_details[track_identifier] = {
                    "track_cache_key": track_key,
                    "details": updated_details,
                }

                singles = self._compute_album_single_set(album_scores)
                previous = self._album_single_sets.get(album_key)
                self._album_single_sets[album_key] = singles
                singles_changed = singles != previous

                if base_score is not None and track_identifier in singles:
                    is_single = True
                    multiplier = 2.0
                    final_score = base_score * multiplier
                elif base_score is not None:
                    final_score = base_score
                else:
                    final_score = None
        else:
            final_score = base_score if base_score is not None else None

        components["is_single"] = is_single
        components["single_multiplier"] = multiplier
        components["final_score"] = final_score

        if album_key and singles_changed:
            self._refresh_album_cache(album_key)

        return final_score, updated_details

    def _refresh_album_cache(self, album_key):
        with self._album_scores_lock:
            album_scores = dict(self._album_track_scores.get(album_key, {}))
            singles = set(self._album_single_sets.get(album_key, set()))
            album_details = dict(self._album_track_details.get(album_key, {}))

        for track_identifier, payload in album_details.items():
            if not isinstance(payload, dict):
                continue
            details = payload.get("details")
            if not isinstance(details, dict):
                continue

            base_score = album_scores.get(track_identifier)
            components = dict(details.get("components") or {})
            components["base_score"] = base_score
            is_single = base_score is not None and track_identifier in singles
            multiplier = 2.0 if is_single else 1.0
            final_score = base_score * multiplier if base_score is not None else None
            components["is_single"] = is_single
            components["single_multiplier"] = multiplier
            components["final_score"] = final_score
            details["components"] = components

            track_cache_key = payload.get("track_cache_key")
            if track_cache_key:
                self._remember_track_cache(track_cache_key, final_score, details)


def chunked(iterable, size):
    """Yield fixed-size chunks from a list."""
    if size <= 0:
        size = 1
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


def resolve_cover_path(cover):
    """Resolve a cover path relative to the playlists file, if needed."""
    if not cover:
        return None

    cover = os.path.expanduser(str(cover).strip())
    if not cover:
        return None

    candidate_paths = []
    if os.path.isabs(cover):
        candidate_paths.append(cover)
    else:
        candidate_paths.append(os.path.join(PLAYLISTS_BASE_DIR, cover))
        candidate_paths.append(os.path.abspath(cover))

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate

    return None


def apply_playlist_cover(playlist_obj, cover):
    """Upload a custom cover image for a playlist if requested."""
    if not playlist_obj or not cover:
        return

    resolved_path = resolve_cover_path(cover)
    if not resolved_path:
        logger.warning(
            "Cover image '%s' was not found on disk; skipping cover upload.",
            cover,
        )
        return

    if not hasattr(playlist_obj, "uploadPoster"):
        logger.debug("Playlist object does not support custom posters; skipping cover upload.")
        return

    try:
        playlist_obj.uploadPoster(filepath=resolved_path)
    except Exception as exc:
        logger.warning(
            "Failed to upload cover '%s' for playlist '%s': %s",
            resolved_path,
            getattr(playlist_obj, "title", "<unknown>"),
            exc,
        )
    else:
        logger.info(
            "Applied custom cover to playlist '%s' from '%s'",
            getattr(playlist_obj, "title", "<unknown>"),
            resolved_path,
        )

def fetch_full_metadata(rating_key):
    """Fetch and cache full metadata for a Plex item (track, album, or artist)."""
    if rating_key is None:
        return None

    key_str = str(rating_key)

    with metadata_cache_lock:
        cached = metadata_cache.get(key_str)
        if cached is not None:
            return cached

    url = f"{PLEX_URL}/library/metadata/{key_str}"
    headers = {"X-Plex-Token": PLEX_TOKEN}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    xml_text = response.text

    with metadata_cache_lock:
        cached = metadata_cache.get(key_str)
        if cached is not None:
            return cached

        metadata_cache[key_str] = xml_text
        _mark_metadata_cache_dirty_locked()

    return xml_text


@lru_cache(maxsize=1024)
def _cached_xml_root(xml_text: str):
    return ET.fromstring(xml_text)


def parse_field_from_xml(xml_text, field):
    """Extract a field (attribute or tag) from Plex XML metadata."""
    try:
        if isinstance(xml_text, bytes):
            xml_text = xml_text.decode("utf-8", errors="ignore")
        elif not isinstance(xml_text, str):
            xml_text = str(xml_text)

        root = _cached_xml_root(xml_text)
        # Plex responses wrap metadata inside a <MediaContainer> root. Different
        # endpoints return different child node names (e.g. ``Directory`` for
        # library lookups, ``Track`` for individual items). Search a list of
        # known element names so we can reuse this helper for all of them.
        node = None
        for candidate in ("Directory", "Track", "Video", "Photo", "Metadata"):
            node = root.find(f"./{candidate}")
            if node is not None:
                break
        if node is None:
            return None

        # Check attributes first
        if field in node.attrib:
            return node.attrib[field]

        # Special handling for tag-based collections like genres/moods/styles
        if field in {"genres", "moods", "styles"}:
            tag_name = {"genres": "Genre", "moods": "Mood", "styles": "Style"}[field]
            tags = []
            for el in root.findall(f".//{tag_name}"):
                raw_tag = el.attrib.get("tag")
                if raw_tag:
                    normalized = raw_tag.strip()
                    if normalized:
                        tags.append(normalized)
            if tags:
                return tags  # return list, no join so downstream can merge intelligently

        # Optionally handle titles or other fields
        if field in ("title", "parentTitle", "grandparentTitle"):
            return node.attrib.get(field)

        return None

    except ET.ParseError:
        logger.debug("XML parse error.")
        return None


def _normalize_metadata_values(raw_value):
    """Normalize Plex metadata values into a flat list of strings."""

    normalized = []
    if raw_value is None:
        return normalized

    if isinstance(raw_value, (list, tuple, set)):
        iterable = raw_value
    else:
        iterable = [raw_value]

    for item in iterable:
        if item is None:
            continue
        if isinstance(item, (list, tuple, set)):
            normalized.extend(_normalize_metadata_values(item))
            continue

        candidate = getattr(item, "tag", item)
        if isinstance(candidate, (list, tuple, set)):
            normalized.extend(_normalize_metadata_values(candidate))
            continue

        text = str(candidate).strip()
        if text:
            normalized.append(text)
        elif candidate in {0, 0.0}:
            normalized.append(str(candidate))

    return normalized


def _extract_metadata_values(source_key, field_name):
    """Return normalized metadata values for a given Plex rating key."""

    if not source_key or not field_name:
        return []

    cache_key = (str(source_key), str(field_name))
    cached = _METADATA_FIELD_CACHE.get(cache_key)
    if cached is not None:
        _touch_cache_entry(_METADATA_FIELD_CACHE, cache_key)
        return list(cached)

    try:
        xml_text = fetch_full_metadata(source_key)
        if not xml_text:
            _set_cache_entry(_METADATA_FIELD_CACHE, cache_key, tuple(), _METADATA_FIELD_CACHE_MAX_SIZE)
            return []
        xml_val = parse_field_from_xml(xml_text, field_name)
    except Exception as exc:
        logger.debug(f"Metadata error for key {source_key}: {exc}")
        _set_cache_entry(_METADATA_FIELD_CACHE, cache_key, tuple(), _METADATA_FIELD_CACHE_MAX_SIZE)
        return []

    normalized = tuple(_normalize_metadata_values(xml_val))
    _set_cache_entry(_METADATA_FIELD_CACHE, cache_key, normalized, _METADATA_FIELD_CACHE_MAX_SIZE)
    return list(normalized)

# ----------------------------
# Attribute Retrieval (with Cache)
# ----------------------------
def get_field_value(track, field):
    """Retrieve and merge a field (e.g., genres) from track, album, and artist levels with caching."""

    cache_key = _build_track_cache_key(track)
    if cache_key is not None:
        cached_bucket = _TRACK_FIELD_CACHE.get(cache_key)
        if cached_bucket is not None:
            _touch_cache_entry(_TRACK_FIELD_CACHE, cache_key)
            cached_values = cached_bucket.get(field)
            if cached_values is not None:
                return list(cached_values)

    values = set()

    resolved_field = FIELD_ALIASES.get(field, field)
    if resolved_field != field:
        field_candidates = [resolved_field]
    else:
        field_candidates = [field]

    merge_styles_for_genres = False
    for candidate in (field, resolved_field):
        if isinstance(candidate, str) and candidate.lower() in {"genre", "genres"}:
            merge_styles_for_genres = True
            break

    # Album type needs to collate information explicitly from album objects
    if field == "album.type":
        parent_key = getattr(track, "parentRatingKey", None)
        if parent_key:
            values.update(_extract_metadata_values(parent_key, "type"))

        parent_type = getattr(track, "parentType", None)
        values.update(_normalize_metadata_values(parent_type))

        album_obj = getattr(track, "album", None)
        album = None
        if callable(album_obj):
            try:
                album = album_obj()
            except TypeError:
                album = None
        else:
            album = album_obj
        if album is not None:
            values.update(_normalize_metadata_values(getattr(album, "type", None)))

        result = sorted(values)
        if cache_key is not None:
            bucket = _TRACK_FIELD_CACHE.get(cache_key)
            if bucket is None:
                bucket = {}
                _set_cache_entry(_TRACK_FIELD_CACHE, cache_key, bucket, _TRACK_FIELD_CACHE_MAX_SIZE)
            else:
                _touch_cache_entry(_TRACK_FIELD_CACHE, cache_key)
            bucket[field] = tuple(result)
        return result

    seen_fields = set()
    style_sources_collected = set()

    def collect_from_candidate(candidate):
        if not candidate or candidate in seen_fields:
            return False
        seen_fields.add(candidate)

        before = len(values)

        # 1️⃣ Try direct object field (most accurate)
        val = getattr(track, candidate, None)
        if val is not None:
            if callable(val):
                try:
                    val = val()
                except TypeError:
                    val = None
            values.update(_normalize_metadata_values(val))

        if merge_styles_for_genres:
            track_styles = getattr(track, "styles", None)
            if track_styles is not None:
                if callable(track_styles):
                    try:
                        track_styles = track_styles()
                    except TypeError:
                        track_styles = None
                values.update(_normalize_metadata_values(track_styles))

        # 2️⃣ Try cached XML for the track
        track_key = getattr(track, "ratingKey", None)
        if track_key:
            values.update(_extract_metadata_values(track_key, candidate))
            if merge_styles_for_genres and "track" not in style_sources_collected:
                values.update(_extract_metadata_values(track_key, "styles"))
                style_sources_collected.add("track")

        # 3️⃣ Try album level
        parent_key = getattr(track, "parentRatingKey", None)
        if parent_key:
            values.update(_extract_metadata_values(parent_key, candidate))
            if merge_styles_for_genres and "album" not in style_sources_collected:
                values.update(_extract_metadata_values(parent_key, "styles"))
                style_sources_collected.add("album")

        # 4️⃣ Try artist level
        artist_key = getattr(track, "grandparentRatingKey", None)
        if artist_key:
            values.update(_extract_metadata_values(artist_key, candidate))
            if merge_styles_for_genres and "artist" not in style_sources_collected:
                values.update(_extract_metadata_values(artist_key, "styles"))
                style_sources_collected.add("artist")

        return len(values) > before

    collected = False
    for candidate in field_candidates:
        if collect_from_candidate(candidate):
            collected = True

    if not collected and resolved_field != field:
        collect_from_candidate(field)

    result = sorted(values)
    if cache_key is not None:
        bucket = _TRACK_FIELD_CACHE.get(cache_key)
        if bucket is None:
            bucket = {}
            _set_cache_entry(_TRACK_FIELD_CACHE, cache_key, bucket, _TRACK_FIELD_CACHE_MAX_SIZE)
        else:
            _touch_cache_entry(_TRACK_FIELD_CACHE, cache_key)
        bucket[field] = tuple(result)

    return result

# ----------------------------
# Field Comparison
# ----------------------------


@dataclass(frozen=True)
class CompiledFilter:
    field: str
    operator: str
    expected: Any
    match_all: bool
    expected_values: Tuple[Any, ...]
    expected_lowers: Tuple[str, ...]
    expected_numeric: Optional[Tuple[Any, ...]]


def _coerce_match_all_flag(value: Any, default: bool = True) -> bool:
    """Coerce truthy/falsey configuration values into a boolean flag."""

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


def _coerce_expected_values(raw_expected: Any) -> Tuple[Any, ...]:
    if isinstance(raw_expected, str) and "," in raw_expected:
        return tuple(segment.strip() for segment in raw_expected.split(","))
    if isinstance(raw_expected, (list, tuple)):
        return tuple(raw_expected)
    if isinstance(raw_expected, set):
        return tuple(raw_expected)
    return (raw_expected,)


def _compile_filter_entry(filter_entry: Any) -> CompiledFilter:
    if isinstance(filter_entry, CompiledFilter):
        return filter_entry

    cached = None
    if isinstance(filter_entry, dict):
        cached = filter_entry.get("_compiled_runtime")
        if isinstance(cached, CompiledFilter):
            return cached

    field = filter_entry.get("field") if isinstance(filter_entry, dict) else getattr(filter_entry, "field", "")
    operator = (
        filter_entry.get("operator") if isinstance(filter_entry, dict) else getattr(filter_entry, "operator", "")
    )
    expected = filter_entry.get("value") if isinstance(filter_entry, dict) else getattr(filter_entry, "expected", None)
    if isinstance(filter_entry, dict):
        match_all_raw = filter_entry.get("match_all")
    else:
        match_all_raw = getattr(filter_entry, "match_all", None)

    match_all = _coerce_match_all_flag(match_all_raw, default=True)

    expected_values = _coerce_expected_values(expected)
    if match_all_raw is None and len(expected_values) > 1:
        # Historically multiple expected values behaved as OR conditions.
        # Preserve that behaviour unless the playlist explicitly opts-in to
        # matching every value via ``match_all``.
        match_all = False
    expected_lowers = tuple(str(value).lower() for value in expected_values)

    numeric_values: Optional[Tuple[Any, ...]] = None
    if operator in {"greater_than", "less_than"}:
        temp_values = []
        for item in expected_values:
            try:
                temp_values.append(float(item))
            except (TypeError, ValueError):
                temp_values.append(None)
        numeric_values = tuple(temp_values)

    compiled = CompiledFilter(
        field=field,
        operator=operator,
        expected=expected,
        match_all=match_all,
        expected_values=expected_values,
        expected_lowers=expected_lowers,
        expected_numeric=numeric_values,
    )

    if isinstance(filter_entry, dict):
        filter_entry["_compiled_runtime"] = compiled

    return compiled


def check_condition(value, operator, expected, match_all=True, compiled: Optional[CompiledFilter] = None):
    """Compare a field value using the given operator."""
    if value is None:
        return False

    if compiled is not None:
        match_all = compiled.match_all
        expected_values = compiled.expected_values
        expected_lowers = compiled.expected_lowers
        expected_numeric = compiled.expected_numeric
    else:
        expected_values = _coerce_expected_values(expected)
        expected_lowers = tuple(str(exp).lower() for exp in expected_values)
        expected_numeric = None
        if operator in {"greater_than", "less_than"}:
            temp_values = []
            for exp_value in expected_values:
                try:
                    temp_values.append(float(exp_value))
                except (TypeError, ValueError):
                    temp_values.append(None)
            expected_numeric = tuple(temp_values)

    values: Iterable[Any]
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]

    results = []

    if operator in {"equals", "does_not_equal", "contains", "does_not_contain"}:
        normalized_values = [str(v).lower() for v in values]
        if operator == "equals":
            for expected_lower in expected_lowers:
                results.append(expected_lower in normalized_values)
        elif operator == "does_not_equal":
            for expected_lower in expected_lowers:
                results.append(all(val != expected_lower for val in normalized_values))
        elif operator == "contains":
            for expected_lower in expected_lowers:
                results.append(any(expected_lower in val for val in normalized_values))
        else:  # does_not_contain
            for expected_lower in expected_lowers:
                results.append(all(expected_lower not in val for val in normalized_values))
    elif operator in {"greater_than", "less_than"}:
        expected_numeric = expected_numeric or tuple(None for _ in expected_values)
        for index, numeric_expected in enumerate(expected_numeric):
            if numeric_expected is None:
                results.append(False)
                continue

            comparison_passed = False
            try:
                for candidate in values:
                    candidate_value = float(candidate)
                    if operator == "greater_than" and candidate_value > numeric_expected:
                        comparison_passed = True
                        break
                    if operator == "less_than" and candidate_value < numeric_expected:
                        comparison_passed = True
                        break
            except (TypeError, ValueError):
                results.append(False)
                continue

            results.append(comparison_passed)
    else:
        logger.warning(f"Unknown operator: {operator}")
        return False

    if operator in {"does_not_equal", "does_not_contain"}:
        # Negative operators should only succeed when *all* expected values fail to match.
        # Treating them as OR conditions (via ``any``) would incorrectly allow matches when
        # one of the disallowed values is present. Instead we require every check to pass.
        return all(results)

    return all(results) if match_all else any(results)

# ----------------------------
# Filter evaluation helpers
# ----------------------------


def _evaluate_track_filters(track, wildcard_filters, regular_filters, log, debug_logging):
    """Determine whether a track should be kept based on wildcard/regular filters."""

    if wildcard_filters and not isinstance(wildcard_filters[0], CompiledFilter):
        wildcard_filters = [_compile_filter_entry(f) for f in wildcard_filters]
    if regular_filters and not isinstance(regular_filters[0], CompiledFilter):
        regular_filters = [_compile_filter_entry(f) for f in regular_filters]

    wildcard_matched = False
    if wildcard_filters:
        for compiled_filter in wildcard_filters:
            field = compiled_filter.field
            operator = compiled_filter.operator
            expected = compiled_filter.expected
            match_all = compiled_filter.match_all

            val = get_field_value(track, field)
            if debug_logging:
                log.debug(
                    "  Wildcard condition: field='%s', operator='%s', expected=%s, match_all=%s",
                    field,
                    operator,
                    expected,
                    match_all,
                )
                log.debug("    Extracted value: %s", val)
            if check_condition(val, operator, expected, match_all, compiled=compiled_filter):
                wildcard_matched = True
                if debug_logging:
                    log.debug(
                        "    ✅ Wildcard condition matched for field '%s'; skipping remaining filters",
                        field,
                    )
                break
            if debug_logging:
                log.debug("    ❌ Wildcard condition failed for field '%s'", field)

    regular_matched = True
    if not wildcard_matched and regular_filters:
        for compiled_filter in regular_filters:
            field = compiled_filter.field
            operator = compiled_filter.operator
            expected = compiled_filter.expected
            match_all = compiled_filter.match_all

            val = get_field_value(track, field)
            if debug_logging:
                log.debug(
                    "  Condition: field='%s', operator='%s', expected=%s, match_all=%s",
                    field,
                    operator,
                    expected,
                    match_all,
                )
                log.debug("    Extracted value: %s", val)
            if not check_condition(val, operator, expected, match_all, compiled=compiled_filter):
                regular_matched = False
                if debug_logging:
                    log.debug("    ❌ Condition failed for field '%s'", field)
                break

    keep = False
    if wildcard_matched:
        keep = True
    elif regular_filters:
        keep = regular_matched
    elif wildcard_filters:
        keep = False
        if debug_logging:
            log.debug("    ❌ Track did not match any wildcard filters")
    else:
        keep = True

    return keep, wildcard_matched

# ----------------------------
# Playlist Builder
# ----------------------------


class FilteringProgressReporter:
    """Emit structured progress updates for playlist filtering."""

    def __init__(self, log_obj, playlist_name, total_tracks):
        self._log = log_obj
        self._playlist = playlist_name
        try:
            total_value = int(total_tracks)
        except (TypeError, ValueError):
            total_value = 0
        self._total = total_value if total_value > 0 else None
        self._start = time.perf_counter()
        self._last_report_time = self._start
        self._last_reported_percent = -1.0
        self._last_reported_count = 0

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _should_report(self, count, percent, now, force):
        if force:
            return True
        if self._total is None:
            if count >= self._last_reported_count + 100:
                return True
        else:
            if percent >= self._last_reported_percent + 1.0:
                return True
        if now - self._last_report_time >= 5.0:
            return True
        return False

    def update(self, count, force=False):
        if count < 0:
            return
        now = time.perf_counter()
        percent = 0.0
        if self._total:
            percent = (float(count) / float(self._total)) * 100.0
        if not self._should_report(count, percent, now, force):
            return

        elapsed = max(0.0, now - self._start)
        rate = (float(count) / elapsed) if elapsed > 0 else 0.0
        eta_seconds = None
        if self._total and rate > 0:
            remaining = max(self._total - count, 0)
            eta_seconds = remaining / rate

        parts = []
        if elapsed >= 0.5:
            parts.append(f"elapsed {self._format_duration(elapsed)}")
        if eta_seconds is not None:
            parts.append(f"ETA {self._format_duration(eta_seconds)}")
        if rate > 0:
            parts.append(f"{rate:.1f} track/s")

        total_text = str(self._total) if self._total else "?"
        message = f"Filtering progress for '{self._playlist}': {count}/{total_text}"
        if self._total:
            message = f"{message} ({percent:.1f}%)"
        if parts:
            message = f"{message} – {' – '.join(parts)}"

        self._log.info(message)
        self._last_reported_count = count
        self._last_report_time = now
        if self._total:
            self._last_reported_percent = percent

    def finalize(self, count):
        self.update(count, force=True)


_SERVER_FILTER_FIELD_MAP: Dict[str, Tuple[str, str]] = {
    # Artist level
    "artist": ("filters", "artist.title"),
    "artist.title": ("filters", "artist.title"),
    "grandparenttitle": ("filters", "artist.title"),
    "artist.id": ("filters", "artist.id"),
    "artist.ratingkey": ("filters", "artist.id"),
    "grandparentratingkey": ("filters", "artist.id"),
    "artist.guid": ("filters", "artist.guid"),
    "grandparentguid": ("filters", "artist.guid"),
    "title": ("filters", "track.title"),
    "track.title": ("filters", "track.title"),
    "track": ("filters", "track.title"),
    "genre": ("filters", "track.genre"),
    "genres": ("filters", "track.genre"),
    "style": ("filters", "track.style"),
    "styles": ("filters", "track.style"),
    "track.style": ("filters", "track.style"),
    "mood": ("filters", "track.mood"),
    "moods": ("filters", "track.mood"),
    "label": ("filters", "track.label"),
    "track.label": ("filters", "track.label"),
    "ratingcount": ("filters", "track.ratingCount"),
    "track.ratingcount": ("filters", "track.ratingCount"),
    "viewcount": ("filters", "track.viewCount"),
    "track.viewcount": ("filters", "track.viewCount"),
    "lastviewedat": ("filters", "track.lastViewedAt"),
    "track.lastviewedat": ("filters", "track.lastViewedAt"),
    "skipcount": ("filters", "track.skipCount"),
    "track.skipcount": ("filters", "track.skipCount"),
    "lastskippedat": ("filters", "track.lastSkippedAt"),
    "track.lastskippedat": ("filters", "track.lastSkippedAt"),
    "userrating": ("filters", "track.userRating"),
    "track.userrating": ("filters", "track.userRating"),
    "lastratedat": ("filters", "track.lastRatedAt"),
    "track.lastratedat": ("filters", "track.lastRatedAt"),
    "addedat": ("filters", "track.addedAt"),
    "track.addedat": ("filters", "track.addedAt"),
    "mediasize": ("filters", "track.mediaSize"),
    "track.mediasize": ("filters", "track.mediaSize"),
    "mediabitrate": ("filters", "track.mediaBitrate"),
    "track.mediabitrate": ("filters", "track.mediaBitrate"),
    "trash": ("filters", "track.trash"),
    "track.trash": ("filters", "track.trash"),
    "location": ("filters", "track.location"),
    "track.location": ("filters", "track.location"),
    "source": ("filters", "track.source"),
    "track.source": ("filters", "track.source"),
    "guid": ("filters", "track.guid"),
    "track.guid": ("filters", "track.guid"),
    "id": ("filters", "track.id"),
    "track.id": ("filters", "track.id"),
    "index": ("filters", "track.index"),
    "track.index": ("filters", "track.index"),
    "updatedat": ("filters", "track.updatedAt"),
    "track.updatedat": ("filters", "track.updatedAt"),
    "duration": ("filters", "track.duration"),
    "track.duration": ("filters", "track.duration"),
    "viewoffset": ("filters", "track.viewOffset"),
    "track.viewoffset": ("filters", "track.viewOffset"),
}


_SERVER_FILTER_TAG_LEVEL_GROUPS: Sequence[Dict[str, Tuple[str, ...]]] = (
    {
        "track.genre": ("album.genre", "album.style"),
        "track.mood": ("album.mood",),
        "track.style": ("album.style",),
    },
    {
        "track.genre": ("artist.genre", "artist.style"),
        "track.mood": ("artist.mood",),
        "track.style": ("artist.style",),
    },
)


def _expand_tag_level_server_filters(
    parameter_sets: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Expand track-level genre/mood/style filters to album and artist level queries."""

    if not parameter_sets:
        return list(parameter_sets)

    expanded: List[Dict[str, Any]] = []

    for params in parameter_sets:
        filters = dict(params.get("filters") or {})
        if not filters:
            expanded.append(dict(params))
            continue

        override_entries: List[Tuple[str, Any, str, str]] = []
        for key, value in filters.items():
            base_key = key[:-1] if key.endswith("&") else key
            suffix = "&" if key.endswith("&") else ""
            for group_mapping in _SERVER_FILTER_TAG_LEVEL_GROUPS:
                if base_key in group_mapping:
                    override_entries.append((key, value, base_key, suffix))
                    break

        if not override_entries:
            expanded.append(dict(params))
            continue

        override_keys = {entry[0] for entry in override_entries}
        base_filters = {k: v for k, v in filters.items() if k not in override_keys}

        level_parameter_sets: List[Dict[str, Any]] = []
        for mapping in _SERVER_FILTER_TAG_LEVEL_GROUPS:
            replacement_options: List[Tuple[str, Any, str, Tuple[str, ...]]] = []
            for original_key, value, base_key, suffix in override_entries:
                replacements = mapping.get(base_key)
                if not replacements:
                    replacement_options = []
                    break
                replacement_options.append((base_key, value, suffix, replacements))

            if not replacement_options:
                continue

            def apply_replacements(
                index: int, current_filters: Dict[str, Any]
            ) -> None:
                if index >= len(replacement_options):
                    combined_filters = dict(base_filters)
                    combined_filters.update(current_filters)
                    new_params = dict(params)
                    if combined_filters:
                        new_params["filters"] = combined_filters
                    else:
                        new_params.pop("filters", None)
                    level_parameter_sets.append(new_params)
                    return

                _, value, suffix, replacements = replacement_options[index]
                for replacement_base in replacements:
                    key_name = f"{replacement_base}{suffix}"
                    next_filters = dict(current_filters)
                    next_filters[key_name] = value
                    apply_replacements(index + 1, next_filters)

            apply_replacements(0, {})

        if level_parameter_sets:
            expanded.append(dict(params))
            expanded.extend(level_parameter_sets)
        else:
            expanded.append(dict(params))

    return expanded


def _normalize_filter_values_for_server(values: Sequence[Any]) -> List[Any]:
    normalized: List[Any] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            normalized.append(stripped)
        else:
            normalized.append(value)
    return normalized


def _build_server_side_search_filters(
    filters: Sequence[CompiledFilter],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Translate simple filters into Plex server-side search parameters."""

    server_kwargs: Dict[str, Any] = {}
    server_filters: Dict[str, Any] = {}
    multi_value_filters: List[Dict[str, Any]] = []

    for compiled in filters:
        operator = (compiled.operator or "").lower()
        if operator not in {"equals", "contains"}:
            continue

        normalized = FIELD_ALIASES.get(compiled.field, compiled.field)
        normalized_lower = str(normalized or "").lower()
        mapping = _SERVER_FILTER_FIELD_MAP.get(normalized_lower)
        if not mapping:
            continue

        values = _normalize_filter_values_for_server(compiled.expected_values)
        if not values:
            continue

        key_type, key_name = mapping
        effective_key = key_name

        if len(values) > 1:
            if compiled.match_all:
                if key_type != "filters":
                    continue
                effective_key = f"{key_name}&"
                payload: Any = list(dict.fromkeys(values))
            else:
                deduped = list(dict.fromkeys(values))
                if not deduped:
                    continue
                conflict = any(
                    entry.get("key_type") == key_type and entry.get("key_name") == key_name
                    for entry in multi_value_filters
                )
                if conflict:
                    return {}, {}, []
                multi_value_filters.append(
                    {"key_type": key_type, "key_name": key_name, "values": deduped}
                )
                continue
        else:
            payload = values[0]

        if len(values) == 1 or compiled.match_all:
            target = server_kwargs if key_type == "kwargs" else server_filters
            if effective_key in target:
                return {}, {}, []
            target[effective_key] = payload

    return server_kwargs, server_filters, multi_value_filters


def _fetch_tracks_with_server_filters(
    library: Any,
    base_kwargs: Dict[str, Any],
    base_filters: Dict[str, Any],
    multi_value_filters: Sequence[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Any], Dict[str, int]]:
    """Execute Plex library searches for the provided server-side filters."""

    def build_search_kwargs(kwargs_part: Dict[str, Any], filters_part: Dict[str, Any]) -> Dict[str, Any]:
        search_kwargs = dict(kwargs_part)
        search_filters = dict(filters_part)
        search_kwargs["libtype"] = "track"
        if search_filters:
            search_kwargs["filters"] = search_filters
        elif "filters" in search_kwargs:
            search_kwargs.pop("filters", None)
        return search_kwargs

    base_kwargs = dict(base_kwargs or {})
    base_filters = dict(base_filters or {})

    search_parameter_sets: List[Dict[str, Any]] = []

    if multi_value_filters:
        def expand(index: int, current_kwargs: Dict[str, Any], current_filters: Dict[str, Any]) -> None:
            if index >= len(multi_value_filters):
                search_parameter_sets.append(build_search_kwargs(current_kwargs, current_filters))
                return

            entry = multi_value_filters[index] or {}
            values = list(entry.get("values") or [])
            if not values:
                expand(index + 1, dict(current_kwargs), dict(current_filters))
                return

            key_type = entry.get("key_type")
            key_name = entry.get("key_name")
            seen_values = []
            for value in values:
                if value in seen_values:
                    continue
                seen_values.append(value)
                next_kwargs = dict(current_kwargs)
                next_filters = dict(current_filters)
                if key_type == "kwargs":
                    next_kwargs[key_name] = value
                else:
                    next_filters[key_name] = value
                expand(index + 1, next_kwargs, next_filters)

        expand(0, dict(base_kwargs), dict(base_filters))
    else:
        search_parameter_sets.append(build_search_kwargs(base_kwargs, base_filters))

    if not search_parameter_sets:
        search_parameter_sets.append(build_search_kwargs(base_kwargs, base_filters))

    search_parameter_sets = _expand_tag_level_server_filters(search_parameter_sets)

    if logger and logger.isEnabledFor(logging.DEBUG):
        for idx, params in enumerate(search_parameter_sets, start=1):
            logger.debug(
                "Server-side filter query %d/%d: %s",
                idx,
                len(search_parameter_sets),
                {k: v for k, v in params.items() if k != "libtype"},
            )

    request_count = len(search_parameter_sets)

    def execute_search(params: Dict[str, Any]) -> List[Any]:
        result = library.search(**params)
        return list(result)

    def log_query_result(index: int, total: int, results: Sequence[Any]) -> None:
        if not logger or not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "Server-side filter query %d/%d returned %d result(s)",
            index,
            total,
            len(results),
        )

    if request_count == 1:
        single_results = execute_search(search_parameter_sets[0])
        log_query_result(1, request_count, single_results)
        results = [single_results]
    else:
        try:
            configured_workers = int(MAX_WORKERS)
        except (TypeError, ValueError):
            configured_workers = 1
        worker_count = max(1, min(request_count, max(configured_workers, 1)))

        results = [None] * request_count  # type: ignore[assignment]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(library.search, **params): idx
                for idx, params in enumerate(search_parameter_sets)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                query_results = list(future.result())
                results[idx] = query_results
                log_query_result(idx + 1, request_count, query_results)

    combined: List[Any] = []
    seen_keys: Set[Any] = set()
    total_before = 0
    duplicates_removed = 0

    def identity(track: Any) -> Any:
        rating_key = getattr(track, "ratingKey", None)
        if rating_key is not None:
            return ("ratingKey", str(rating_key))
        guid = getattr(track, "guid", None)
        if guid:
            return ("guid", str(guid))
        return ("id", id(track))

    for track_list in results:
        if not track_list:
            continue
        for track in track_list:
            total_before += 1
            key = identity(track)
            if key in seen_keys:
                duplicates_removed += 1
                continue
            seen_keys.add(key)
            combined.append(track)

    stats = {
        "requests": request_count,
        "original_count": total_before,
        "duplicates_removed": duplicates_removed,
    }

    return combined, stats



def _determine_metadata_levels_for_filters(filters: Sequence[CompiledFilter]) -> Set[str]:
    """Return which Plex objects (track/album/artist) require metadata prefetching."""

    levels: Set[str] = set()
    for compiled in filters:
        field = (compiled.field or "").strip()
        if not field:
            continue

        normalized = FIELD_ALIASES.get(field, field) or ""
        lower_field = field.lower()
        normalized_lower = str(normalized).lower()

        levels.add("track")

        if field == "album.type" or lower_field.startswith("album.") or normalized_lower.startswith("parent"):
            levels.add("album")

        if lower_field.startswith("artist.") or normalized_lower.startswith("grandparent"):
            levels.add("artist")

    return levels


def _warm_metadata_cache(
    tracks: Sequence[Any],
    wildcard_filters: Sequence[CompiledFilter],
    regular_filters: Sequence[CompiledFilter],
    log,
) -> Tuple[int, int]:
    """Pre-fetch Plex XML metadata required by the configured filters."""

    compiled_filters: List[CompiledFilter] = []
    if regular_filters:
        compiled_filters.extend(regular_filters)
    if wildcard_filters:
        compiled_filters.extend(wildcard_filters)

    if not compiled_filters or not tracks:
        return 0, 0

    levels = _determine_metadata_levels_for_filters(compiled_filters)
    if not levels:
        return 0, 0

    rating_keys: Set[str] = set()
    for track in tracks:
        if track is None:
            continue
        if "track" in levels:
            track_key = getattr(track, "ratingKey", None)
            if track_key:
                rating_keys.add(str(track_key))
        if "album" in levels:
            album_key = getattr(track, "parentRatingKey", None)
            if album_key:
                rating_keys.add(str(album_key))
        if "artist" in levels:
            artist_key = getattr(track, "grandparentRatingKey", None)
            if artist_key:
                rating_keys.add(str(artist_key))

    if not rating_keys:
        return 0, 0

    with metadata_cache_lock:
        pending = [key for key in rating_keys if key not in metadata_cache]

    if not pending:
        return 0, 0

    try:
        configured_workers = int(MAX_WORKERS)
    except (TypeError, ValueError):
        configured_workers = 1

    worker_count = max(1, min(len(pending), max(configured_workers, 1)))

    failures = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(fetch_full_metadata, key): key for key in pending}
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - network errors handled gracefully
                failures += 1
                if log.isEnabledFor(logging.DEBUG):
                    log.debug("Failed to pre-fetch metadata for ratingKey=%s: %s", key, exc)

    return len(pending), failures


def _sort_tracks_in_place(
    tracks,
    resolved_sort_by,
    sort_desc,
    log,
    dedup_popularity_cache,
    album_popularity_cache,
    album_popularity_cache_by_object,
    *,
    debug_logging=False,
):
    if not tracks or not resolved_sort_by:
        return 0.0

    sort_value_cache: Dict[str, Any] = {}
    sort_value_cache_by_object: Dict[int, Any] = {}

    def _get_sort_value(track):
        cache_key = getattr(track, "ratingKey", None)
        cache_key_str = str(cache_key) if cache_key is not None else None
        object_cache_key = id(track)
        if cache_key_str and cache_key_str in sort_value_cache:
            return sort_value_cache[cache_key_str]

        if cache_key_str is None and object_cache_key in sort_value_cache_by_object:
            return sort_value_cache_by_object[object_cache_key]

        if resolved_sort_by == "__rating_count__":
            popularity_value, _ = _resolve_popularity_for_sort(
                track,
                cache_key_str,
                object_cache_key,
                dedup_popularity_cache,
                album_popularity_cache,
                album_popularity_cache_by_object,
                playlist_logger=log,
                sort_desc=sort_desc,
            )

            if cache_key_str:
                sort_value_cache[cache_key_str] = popularity_value
            else:
                sort_value_cache_by_object[object_cache_key] = popularity_value
            return popularity_value

        if resolved_sort_by == "__album_year__":
            year_token = _resolve_album_year(track)
            if year_token is None:
                sort_value = None
            else:
                try:
                    numeric_year = int(str(year_token))
                except (TypeError, ValueError):
                    numeric_year = year_token

                release_date = _resolve_album_release_date(track)
                if release_date is None:
                    release_component = (
                        datetime.date.max if not sort_desc else datetime.date.min
                    )
                else:
                    release_component = release_date

                sort_value = (
                    numeric_year,
                    release_component,
                    getattr(track, "title", ""),
                    getattr(track, "grandparentTitle", ""),
                )

            if cache_key_str:
                sort_value_cache[cache_key_str] = sort_value
            else:
                sort_value_cache_by_object[object_cache_key] = sort_value
            return sort_value

        if resolved_sort_by == "__alphabetical__":
            raw_title = getattr(track, "title", "") or ""
            raw_artist = getattr(track, "grandparentTitle", "") or ""
            normalized_title = unicodedata.normalize("NFKD", str(raw_title)).casefold()
            normalized_artist = unicodedata.normalize("NFKD", str(raw_artist)).casefold()
            sort_value = (normalized_title, normalized_artist, str(raw_title))
            if cache_key_str:
                sort_value_cache[cache_key_str] = sort_value
            else:
                sort_value_cache_by_object[object_cache_key] = sort_value
            return sort_value

        if resolved_sort_by in {"ratingCount", "parentRatingCount"}:
            if debug_logging:
                log.debug(
                    "Evaluating %s for track '%s' (album='%s', ratingKey=%s)",
                    resolved_sort_by,
                    getattr(track, "title", "<unknown>"),
                    getattr(track, "parentTitle", "<unknown>"),
                    cache_key,
                )

            def _coerce_numeric(value):
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    stripped = value.strip()
                    if not stripped:
                        return None
                    try:
                        return float(stripped)
                    except ValueError:
                        return None
                return None

            direct_value = getattr(track, resolved_sort_by, None)
            if callable(direct_value):
                try:
                    direct_value = direct_value()
                except TypeError:
                    direct_value = None

            direct_numeric = _coerce_numeric(direct_value)

            if direct_numeric is not None:
                if debug_logging:
                    log.debug(
                        "Chosen %s=%s for track '%s'",
                        resolved_sort_by,
                        direct_numeric,
                        getattr(track, "title", "<unknown>"),
                    )
                if cache_key_str:
                    sort_value_cache[cache_key_str] = direct_numeric
                return direct_numeric

        value = getattr(track, resolved_sort_by, None)
        if value is not None:
            if isinstance(value, str):
                stripped_value = value.strip()
                if stripped_value:
                    try:
                        value = float(stripped_value)
                    except ValueError:
                        pass
            if cache_key_str:
                sort_value_cache[cache_key_str] = value
            return value

        try:
            xml_text = fetch_full_metadata(track.ratingKey)
        except Exception as exc:  # pragma: no cover - network errors handled earlier
            log.debug(
                "Failed to fetch metadata for sorting field '%s' on ratingKey=%s: %s",
                resolved_sort_by,
                getattr(track, "ratingKey", "<no-key>"),
                exc,
            )
            if cache_key_str:
                sort_value_cache[cache_key_str] = None
            return None

        xml_value = parse_field_from_xml(xml_text, resolved_sort_by)

        if isinstance(xml_value, (list, set, tuple)):
            xml_value = next(iter(xml_value), None)

        if isinstance(xml_value, str):
            stripped = xml_value.strip()
            if stripped:
                try:
                    if cache_key_str:
                        sort_value_cache[cache_key_str] = float(stripped)
                        return sort_value_cache[cache_key_str]
                except ValueError:
                    if cache_key_str:
                        sort_value_cache[cache_key_str] = stripped
                        return sort_value_cache[cache_key_str]
            if cache_key_str:
                sort_value_cache[cache_key_str] = None
            return None

        if isinstance(xml_value, (int, float)):
            if cache_key_str:
                sort_value_cache[cache_key_str] = float(xml_value)
                return sort_value_cache[cache_key_str]

        if cache_key_str:
            sort_value_cache[cache_key_str] = xml_value
        return xml_value

    def _normalize_sort_value(value):
        if value is None:
            return None
        if isinstance(value, tuple):
            return tuple(_normalize_sort_value(item) for item in value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return value.lower()
        if hasattr(value, "timestamp"):
            try:
                return float(value.timestamp())
            except Exception:  # pragma: no cover - defensive
                pass
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:  # pragma: no cover - defensive
                pass
        return str(value)

    def _compare_tracks(left, right):
        left_val = _get_sort_value(left)
        right_val = _get_sort_value(right)

        if left_val is None and right_val is None:
            return 0
        if left_val is None:
            return 1
        if right_val is None:
            return -1

        left_key = _normalize_sort_value(left_val)
        right_key = _normalize_sort_value(right_val)

        try:
            if left_key < right_key:
                result = -1
            elif left_key > right_key:
                result = 1
            else:
                result = 0
        except TypeError:
            left_key = str(left_key)
            right_key = str(right_key)
            if left_key < right_key:
                result = -1
            elif left_key > right_key:
                result = 1
            else:
                result = 0

        return -result if sort_desc else result

    sort_start = time.perf_counter()
    tracks.sort(key=cmp_to_key(_compare_tracks))
    return time.perf_counter() - sort_start


def _run_spotify_playlist_build(
    name,
    config,
    log,
    plex_server,
    library,
    spotify_url,
    resolved_sort_by,
    sort_desc,
    resolved_after_sort,
    after_sort_desc,
    chunk_size,
    top_5_boost_value,
    cover_path,
    build_start,
    limit,
):
    fetch_start = time.perf_counter()
    if not spotify_url:
        log.warning(
            "Spotify playlist '%s' does not have a URL configured; no tracks will be added.",
            name,
        )
        matched_tracks: List[Any] = []
        spotify_stats = {
            "normalized_url": "",
            "total_tracks": 0,
            "matched_tracks": 0,
            "unmatched_tracks": 0,
        }
    else:
        try:
            matched_tracks, spotify_stats = _collect_spotify_tracks(
                spotify_url,
                library,
                log,
            )
        except Exception as exc:
            log.error(
                "Failed to load Spotify playlist for '%s' from '%s': %s",
                name,
                spotify_url,
                exc,
            )
            matched_tracks = []
            spotify_stats = {
                "normalized_url": spotify_url,
                "total_tracks": 0,
                "matched_tracks": 0,
                "unmatched_tracks": 0,
            }

    fetch_duration = time.perf_counter() - fetch_start
    total_tracks = spotify_stats.get("total_tracks", 0)
    matched_total = len(matched_tracks)
    normalized_url = spotify_stats.get("normalized_url")

    if normalized_url:
        log.info(
            "Matched %s of %s track(s) from Spotify playlist %s",
            matched_total,
            total_tracks,
            normalized_url,
        )
    else:
        log.info(
            "Matched %s of %s track(s) from Spotify playlist '%s'",
            matched_total,
            total_tracks,
            name,
        )

    unmatched = spotify_stats.get("unmatched_tracks", 0)
    if unmatched:
        log.warning(
            "Unable to match %s Spotify track(s) for playlist '%s'",
            unmatched,
            name,
        )

    try:
        existing = plex_server.playlist(name)
    except Exception:
        existing = None

    deleted_existing = False
    playlist_obj = None
    playlist_update_duration = 0.0
    debug_logging = log.isEnabledFor(logging.DEBUG)

    filter_duration = 0.0
    filter_rate = 0.0
    warm_duration = 0.0
    dedup_duration = 0.0
    sort_duration = 0.0

    dedup_popularity_cache: Dict[str, Any] = {}
    if matched_tracks:
        dedup_start = time.perf_counter()
        matched_tracks, dedup_popularity_cache, duplicates_removed = _deduplicate_tracks(
            matched_tracks,
            log,
        )
        dedup_duration = time.perf_counter() - dedup_start
        if duplicates_removed:
            log.info(
                "Removed %s duplicate track(s) from playlist '%s' after popularity comparison",
                duplicates_removed,
                name,
            )

    album_popularity_cache: Dict[str, Any] = {}
    album_popularity_cache_by_object: Dict[int, Any] = {}
    needs_popularity_sort = (
        resolved_sort_by == "__rating_count__" or resolved_after_sort == "__rating_count__"
    )
    if needs_popularity_sort and matched_tracks:
        (
            album_popularity_cache,
            album_popularity_cache_by_object,
        ) = _compute_album_popularity_boosts(
            matched_tracks,
            dedup_popularity_cache,
            playlist_logger=log,
            top_5_boost=top_5_boost_value,
        )

    match_count = len(matched_tracks)

    if matched_tracks:
        sort_duration += _sort_tracks_in_place(
            matched_tracks,
            resolved_sort_by,
            sort_desc,
            log,
            dedup_popularity_cache,
            album_popularity_cache,
            album_popularity_cache_by_object,
            debug_logging=debug_logging,
        )

    artist_limit_raw = config.get("artist_limit")
    if artist_limit_raw is not None and matched_tracks:
        try:
            artist_limit_value = int(artist_limit_raw)
        except (TypeError, ValueError):
            log.warning(
                "Invalid artist_limit '%s' for playlist '%s'; expected integer.",
                artist_limit_raw,
                name,
            )
        else:
            if artist_limit_value > 0:
                artist_counts = {}
                limited_tracks = []
                removed_for_limit = 0

                for track in matched_tracks:
                    artist_name = _normalize_compare_value(getattr(track, "grandparentTitle", None))
                    if not artist_name:
                        fallback_artist = (
                            getattr(track, "grandparentRatingKey", None)
                            or getattr(track, "grandparentGuid", None)
                            or getattr(track, "originalTitle", None)
                        )
                        artist_name = _normalize_compare_value(fallback_artist) or "__unknown__"

                    current_count = artist_counts.get(artist_name, 0)
                    if current_count >= artist_limit_value:
                        removed_for_limit += 1
                        if debug_logging:
                            log.debug(
                                "Skipping track '%s' by '%s' due to artist limit %s",
                                getattr(track, "title", "<unknown>"),
                                getattr(track, "grandparentTitle", "<unknown>"),
                                artist_limit_value,
                            )
                        continue

                    artist_counts[artist_name] = current_count + 1
                    limited_tracks.append(track)

                if removed_for_limit:
                    log.info(
                        "Applied artist limit (%s) for playlist '%s' – removed %s track(s)",
                        artist_limit_value,
                        name,
                        removed_for_limit,
                    )

                matched_tracks = limited_tracks
                match_count = len(matched_tracks)
            else:
                log.warning(
                    "Artist limit for playlist '%s' must be positive; received %s.",
                    name,
                    artist_limit_raw,
                )

    album_limit_raw = config.get("album_limit")
    if album_limit_raw is not None and matched_tracks:
        try:
            album_limit_value = int(album_limit_raw)
        except (TypeError, ValueError):
            log.warning(
                "Invalid album_limit '%s' for playlist '%s'; expected integer.",
                album_limit_raw,
                name,
            )
        else:
            if album_limit_value > 0:
                album_counts = {}
                limited_tracks = []
                removed_for_album_limit = 0

                for track in matched_tracks:
                    album_name = _normalize_compare_value(getattr(track, "parentTitle", None))
                    if not album_name:
                        fallback_album = (
                            getattr(track, "parentRatingKey", None)
                            or getattr(track, "parentGuid", None)
                        )
                        album_name = _normalize_compare_value(fallback_album) or "__unknown_album__"

                    current_count = album_counts.get(album_name, 0)
                    if current_count >= album_limit_value:
                        removed_for_album_limit += 1
                        if debug_logging:
                            log.debug(
                                "Skipping track '%s' from album '%s' due to album limit %s",
                                getattr(track, "title", "<unknown>"),
                                getattr(track, "parentTitle", "<unknown>"),
                                album_limit_value,
                            )
                        continue

                    album_counts[album_name] = current_count + 1
                    limited_tracks.append(track)

                if removed_for_album_limit:
                    log.info(
                        "Applied album limit (%s) for playlist '%s' – removed %s track(s)",
                        album_limit_value,
                        name,
                        removed_for_album_limit,
                    )

                matched_tracks = limited_tracks
                match_count = len(matched_tracks)
            else:
                log.warning(
                    "Album limit for playlist '%s' must be positive; received %s.",
                    name,
                    album_limit_raw,
                )

    if limit and matched_tracks:
        matched_tracks = matched_tracks[:limit]
        match_count = len(matched_tracks)

    if matched_tracks:
        after_sort_duration = _sort_tracks_in_place(
            matched_tracks,
            resolved_after_sort,
            after_sort_desc,
            log,
            dedup_popularity_cache,
            album_popularity_cache,
            album_popularity_cache_by_object,
            debug_logging=debug_logging,
        )
        sort_duration += after_sort_duration

    log.info(f"Playlist '{name}' → {match_count} matching tracks")

    if existing and not deleted_existing:
        try:
            existing.delete()
        except Exception as exc:
            log.warning(f"Failed to delete existing playlist '{name}': {exc}")
        else:
            deleted_existing = True

    if not matched_tracks:
        total_duration = time.perf_counter() - build_start
        log.info(
            "Performance summary for '%s': fetch=%.2fs, prefetch=%.2fs, filter=%.2fs (%.1f track/s), dedup=%.2fs, sort=%.2fs, update=%.2fs, total=%.2fs",
            name,
            fetch_duration,
            warm_duration,
            filter_duration,
            filter_rate,
            dedup_duration,
            sort_duration,
            playlist_update_duration,
            total_duration,
        )
        log.info(f"✅ Finished building '{name}' (0 tracks)")
        return

    update_start = time.perf_counter()
    for chunk in chunked(matched_tracks, chunk_size):
        try:
            if playlist_obj is None:
                playlist_obj = plex_server.createPlaylist(name, items=chunk)
            else:
                playlist_obj.addItems(chunk)
        except Exception as exc:
            log.error(f"Failed to update playlist '{name}': {exc}")
            raise
    playlist_update_duration += time.perf_counter() - update_start

    apply_playlist_cover(playlist_obj, cover_path)
    total_duration = time.perf_counter() - build_start
    update_rate = (match_count / playlist_update_duration) if playlist_update_duration > 0 else 0.0
    log.info(
        "Performance summary for '%s': fetch=%.2fs, prefetch=%.2fs, filter=%.2fs (%.1f track/s), dedup=%.2fs, sort=%.2fs, update=%.2fs (%.1f track/s), total=%.2fs",
        name,
        fetch_duration,
        warm_duration,
        filter_duration,
        filter_rate,
        dedup_duration,
        sort_duration,
        playlist_update_duration,
        update_rate,
        total_duration,
    )
    log.info(f"✅ Finished building '{name}' ({match_count} tracks)")


def _run_playlist_build(name, config, log, playlist_handler, playlist_log_path):
    if playlist_handler:
        log.debug(
            "Per-playlist debug logging for '%s' → %s",
            name,
            playlist_log_path,
        )

    log.info(f"Building playlist: {name}")
    build_start = time.perf_counter()
    playlist_source = _normalize_playlist_source(config.get("source"))
    spotify_url = ""
    if playlist_source == "spotify":
        raw_spotify_url = config.get("spotify_url")
        spotify_url = str(raw_spotify_url).strip() if raw_spotify_url else ""
    filters = config.get("plex_filter", [])
    boost_rules = config.get("popularity_boosts", []) or []
    wildcard_filters = [_compile_filter_entry(f) for f in filters if bool(f.get("wildcard"))]
    regular_filters = [_compile_filter_entry(f) for f in filters if not bool(f.get("wildcard"))]
    plex_server = get_plex_server()
    library = plex_server.library.section(config.get("library", LIBRARY_NAME))
    limit = config.get("limit")
    sort_by = _normalize_sort_option(config.get("sort_by"))
    after_sort = _normalize_sort_option(config.get("after_sort"))
    cover_path = config.get("cover")
    resolved_sort_by = sort_by
    resolved_after_sort = after_sort
    sort_desc_in_config = "sort_desc" in config
    sort_desc = config.get("sort_desc", True)
    after_sort_desc_in_config = "after_sort_desc" in config
    after_sort_desc = config.get("after_sort_desc", True)
    if sort_by == "popularity":
        resolved_sort_by = "__rating_count__"
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Sort field '%s' will use Plex rating counts", sort_by)

    if sort_by == "alphabetical":
        resolved_sort_by = "__alphabetical__"
        if not sort_desc_in_config:
            sort_desc = False
    elif sort_by == "reverse_alphabetical":
        resolved_sort_by = "__alphabetical__"
        if not sort_desc_in_config:
            sort_desc = True
    elif sort_by == "oldest_first":
        resolved_sort_by = "__album_year__"
        if not sort_desc_in_config:
            sort_desc = False
    elif sort_by == "newest_first":
        resolved_sort_by = "__album_year__"
        if not sort_desc_in_config:
            sort_desc = True
    if after_sort == "popularity":
        resolved_after_sort = "__rating_count__"
    if after_sort == "alphabetical":
        resolved_after_sort = "__alphabetical__"
        if not after_sort_desc_in_config:
            after_sort_desc = False
    elif after_sort == "reverse_alphabetical":
        resolved_after_sort = "__alphabetical__"
        if not after_sort_desc_in_config:
            after_sort_desc = True
    elif after_sort == "oldest_first":
        resolved_after_sort = "__album_year__"
        if not after_sort_desc_in_config:
            after_sort_desc = False
    elif after_sort == "newest_first":
        resolved_after_sort = "__album_year__"
        if not after_sort_desc_in_config:
            after_sort_desc = True
    chunk_size = config.get("chunk_size", PLAYLIST_CHUNK_SIZE)
    top_5_boost_raw = config.get("top_5_boost", 1.0)
    top_5_boost_value = _coerce_non_negative_float(top_5_boost_raw)
    if top_5_boost_value is None:
        top_5_boost_value = 1.0
    if playlist_source == "spotify":
        _run_spotify_playlist_build(
            name,
            config,
            log,
            plex_server,
            library,
            spotify_url,
            resolved_sort_by,
            sort_desc,
            resolved_after_sort,
            after_sort_desc,
            chunk_size,
            top_5_boost_value,
            cover_path,
            build_start,
            limit,
        )
        return
    stream_requested = config.get("stream_while_filtering", False)
    stream_enabled = stream_requested and not sort_by and not limit
    if stream_enabled:
        disable_reasons = []
        artist_limit_raw = config.get("artist_limit")
        album_limit_raw = config.get("album_limit")

        if isinstance(artist_limit_raw, str):
            artist_limit_normalized = artist_limit_raw.strip()
        else:
            artist_limit_normalized = artist_limit_raw

        if isinstance(album_limit_raw, str):
            album_limit_normalized = album_limit_raw.strip()
        else:
            album_limit_normalized = album_limit_raw

        if artist_limit_normalized not in (None, ""):
            disable_reasons.append("artist limits")
        if album_limit_normalized not in (None, ""):
            disable_reasons.append("album limits")

        if disable_reasons:
            reason_text = ", ".join(disable_reasons)
            if log.isEnabledFor(logging.INFO):
                log.info(
                    "stream_while_filtering requested for '%s' but disabled because %s require full post-filter processing.",
                    name,
                    reason_text,
                )
            stream_enabled = False
        elif log.isEnabledFor(logging.INFO):
            log.info(
                "stream_while_filtering enabled for '%s'; deduplication (and any artist/album limits) will be skipped.",
                name,
            )

    server_kwargs, server_filter_dict, multi_value_filters = _build_server_side_search_filters(regular_filters)

    fetch_start = time.perf_counter()
    fetch_stats = None
    if server_kwargs or server_filter_dict or multi_value_filters:
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "Fetching tracks for '%s' with server-side filters: kwargs=%s, filters=%s, multi_value=%s",
                name,
                server_kwargs,
                server_filter_dict,
                multi_value_filters,
            )
        all_tracks, fetch_stats = _fetch_tracks_with_server_filters(
            library,
            server_kwargs,
            server_filter_dict,
            multi_value_filters,
            logger=log,
        )
    else:
        all_tracks = library.searchTracks()

    total_tracks = len(all_tracks)
    fetch_duration = time.perf_counter() - fetch_start

    if fetch_stats:
        meta_parts: List[str] = []
        request_count = fetch_stats.get("requests", 0)
        duplicates_removed = fetch_stats.get("duplicates_removed", 0)
        if request_count:
            plural = "query" if request_count == 1 else "queries"
            meta_parts.append(f"using server-side filters across {request_count} {plural}")
        else:
            meta_parts.append("using server-side filters")
        if duplicates_removed:
            plural = "track" if duplicates_removed == 1 else "tracks"
            meta_parts.append(f"removed {duplicates_removed} duplicate {plural}")
        meta_suffix = f" ({'; '.join(meta_parts)})" if meta_parts else ""
        log.info(
            "Fetched %s tracks from %s in %.2fs%s",
            total_tracks,
            config.get("library", LIBRARY_NAME),
            fetch_duration,
            meta_suffix,
        )
    else:
        log.info(
            "Fetched %s tracks from %s in %.2fs",
            total_tracks,
            config.get("library", LIBRARY_NAME),
            fetch_duration,
        )

    warm_requests = warm_failures = 0
    warm_duration = 0.0
    if total_tracks and (regular_filters or wildcard_filters):
        warm_start = time.perf_counter()
        warm_requests, warm_failures = _warm_metadata_cache(
            all_tracks,
            wildcard_filters,
            regular_filters,
            log,
        )
        if warm_requests:
            warm_duration = time.perf_counter() - warm_start
            successful = warm_requests - warm_failures
            if warm_failures:
                log.warning(
                    "Prefetched %d/%d metadata entries for '%s' in %.2fs (failures=%d)",
                    successful,
                    warm_requests,
                    name,
                    warm_duration,
                    warm_failures,
                )
            else:
                log.info(
                    "Prefetched %d metadata entries for '%s' in %.2fs",
                    successful,
                    name,
                    warm_duration,
                )
        elif log.isEnabledFor(logging.DEBUG):
            log.debug("Metadata cache already warm for '%s'", name)

    progress_reporter = FilteringProgressReporter(log, name, total_tracks)

    try:
        existing = plex_server.playlist(name)
    except Exception:
        existing = None
    deleted_existing = False

    matched_tracks = []
    stream_buffer = []
    playlist_obj = None
    match_count = 0
    debug_logging = log.isEnabledFor(logging.DEBUG)
    playlist_update_duration = 0.0

    def flush_stream_buffer():
        nonlocal playlist_obj, stream_buffer, deleted_existing, playlist_update_duration
        if not stream_buffer:
            return
        if existing and not deleted_existing:
            try:
                existing.delete()
            except Exception as exc:
                log.warning(f"Failed to delete existing playlist '{name}': {exc}")
            deleted_existing = True
        flush_start = time.perf_counter()
        try:
            if playlist_obj is None:
                playlist_obj = plex_server.createPlaylist(name, items=list(stream_buffer))
            else:
                playlist_obj.addItems(list(stream_buffer))
        except Exception as exc:
            log.error(f"Failed to update playlist '{name}': {exc}")
            raise
        finally:
            stream_buffer.clear()
            playlist_update_duration += time.perf_counter() - flush_start

    filter_start = time.perf_counter()
    processed_count = 0
    with tqdm(total=total_tracks, desc=f"Filtering '{name}'", unit="track", dynamic_ncols=True) as pbar:
        for track in all_tracks:
            if debug_logging:
                track_title = getattr(track, "title", "<unknown title>")
                track_artist = getattr(track, "grandparentTitle", "<unknown artist>")
                track_album = getattr(track, "parentTitle", "<unknown album>")
                log.debug(
                    "Evaluating track '%s' by '%s' on '%s' (ratingKey=%s)",
                    track_title,
                    track_artist,
                    track_album,
                    getattr(track, "ratingKey", "<no-key>")
                )

            keep, wildcard_matched = _evaluate_track_filters(
                track,
                wildcard_filters,
                regular_filters,
                log,
                debug_logging,
            )

            if keep:
                match_count += 1
                if stream_enabled:
                    stream_buffer.append(track)
                    if len(stream_buffer) >= chunk_size:
                        flush_stream_buffer()
                else:
                    matched_tracks.append(track)
                if debug_logging:
                    if wildcard_matched:
                        log.debug("    ✅ Track kept due to wildcard filter match")
                    elif regular_filters:
                        log.debug("    ✅ Track matched all non-wildcard conditions")
                    else:
                        log.debug("    ✅ Track kept (no filters defined)")
            pbar.update(1)
            processed_count += 1
            progress_reporter.update(processed_count)

    filter_duration = time.perf_counter() - filter_start
    filter_rate = (total_tracks / filter_duration) if filter_duration > 0 else 0.0

    progress_reporter.finalize(processed_count)

    if stream_enabled:
        flush_stream_buffer()
        if not deleted_existing and existing:
            try:
                existing.delete()
            except Exception as exc:
                log.warning(f"Failed to delete existing playlist '{name}': {exc}")
            deleted_existing = True
        apply_playlist_cover(playlist_obj, cover_path)
        log.info(f"Playlist '{name}' → {match_count} matching tracks")
        if match_count == 0:
            log.info(f"No tracks matched for '{name}'. Playlist will not be recreated.")
        total_duration = time.perf_counter() - build_start
        log.info(
            "Performance summary for '%s': fetch=%.2fs, prefetch=%.2fs, filter=%.2fs (%.1f track/s), update=%.2fs, total=%.2fs",
            name,
            fetch_duration,
            warm_duration,
            filter_duration,
            filter_rate,
            playlist_update_duration,
            total_duration,
        )
        log.info(f"✅ Finished building '{name}' ({match_count} tracks)")
        return

    dedup_popularity_cache = {}
    dedup_duration = 0.0
    if matched_tracks:
        dedup_start = time.perf_counter()
        matched_tracks, dedup_popularity_cache, duplicates_removed = _deduplicate_tracks(
            matched_tracks,
            log,
        )
        dedup_duration = time.perf_counter() - dedup_start
        if duplicates_removed:
            log.info(
                "Removed %s duplicate track(s) from playlist '%s' after popularity comparison",
                duplicates_removed,
                name,
            )

    album_popularity_cache: Dict[str, Any] = {}
    album_popularity_cache_by_object: Dict[int, Any] = {}
    needs_popularity_sort = (
        resolved_sort_by == "__rating_count__" or resolved_after_sort == "__rating_count__"
    )
    if needs_popularity_sort:
        (
            album_popularity_cache,
            album_popularity_cache_by_object,
        ) = _compute_album_popularity_boosts(
            matched_tracks,
            dedup_popularity_cache,
            playlist_logger=log,
            top_5_boost=top_5_boost_value,
        )
    if boost_rules:
        _apply_configured_popularity_boosts(
            matched_tracks,
            boost_rules,
            dedup_popularity_cache,
            album_popularity_cache if needs_popularity_sort else {},
            album_popularity_cache_by_object if needs_popularity_sort else {},
            playlist_logger=log,
        )
    match_count = len(matched_tracks)

    sort_duration = _sort_tracks_in_place(
        matched_tracks,
        resolved_sort_by,
        sort_desc,
        log,
        dedup_popularity_cache,
        album_popularity_cache,
        album_popularity_cache_by_object,
        debug_logging=debug_logging,
    )
    artist_limit_raw = config.get("artist_limit")
    if artist_limit_raw is not None:
        try:
            artist_limit_value = int(artist_limit_raw)
        except (TypeError, ValueError):
            log.warning(
                "Invalid artist_limit '%s' for playlist '%s'; expected integer.",
                artist_limit_raw,
                name,
            )
        else:
            if artist_limit_value > 0:
                artist_counts = {}
                limited_tracks = []
                removed_for_limit = 0

                for track in matched_tracks:
                    artist_name = _normalize_compare_value(getattr(track, "grandparentTitle", None))
                    if not artist_name:
                        fallback_artist = (
                            getattr(track, "grandparentRatingKey", None)
                            or getattr(track, "grandparentGuid", None)
                            or getattr(track, "originalTitle", None)
                        )
                        artist_name = _normalize_compare_value(fallback_artist) or "__unknown__"

                    current_count = artist_counts.get(artist_name, 0)
                    if current_count >= artist_limit_value:
                        removed_for_limit += 1
                        if debug_logging:
                            log.debug(
                                "Skipping track '%s' by '%s' due to artist limit %s",
                                getattr(track, "title", "<unknown>"),
                                getattr(track, "grandparentTitle", "<unknown>"),
                                artist_limit_value,
                            )
                        continue

                    artist_counts[artist_name] = current_count + 1
                    limited_tracks.append(track)

                if removed_for_limit:
                    log.info(
                        "Applied artist limit (%s) for playlist '%s' – removed %s track(s)",
                        artist_limit_value,
                        name,
                        removed_for_limit,
                    )

                matched_tracks = limited_tracks
                match_count = len(matched_tracks)
            else:
                log.warning(
                    "Artist limit for playlist '%s' must be positive; received %s.",
                    name,
                    artist_limit_raw,
                )

    album_limit_raw = config.get("album_limit")
    if album_limit_raw is not None:
        try:
            album_limit_value = int(album_limit_raw)
        except (TypeError, ValueError):
            log.warning(
                "Invalid album_limit '%s' for playlist '%s'; expected integer.",
                album_limit_raw,
                name,
            )
        else:
            if album_limit_value > 0:
                album_counts = {}
                limited_tracks = []
                removed_for_album_limit = 0

                for track in matched_tracks:
                    album_name = _normalize_compare_value(getattr(track, "parentTitle", None))
                    if not album_name:
                        fallback_album = (
                            getattr(track, "parentRatingKey", None)
                            or getattr(track, "parentGuid", None)
                        )
                        album_name = _normalize_compare_value(fallback_album) or "__unknown_album__"

                    current_count = album_counts.get(album_name, 0)
                    if current_count >= album_limit_value:
                        removed_for_album_limit += 1
                        if debug_logging:
                            log.debug(
                                "Skipping track '%s' from album '%s' due to album limit %s",
                                getattr(track, "title", "<unknown>"),
                                getattr(track, "parentTitle", "<unknown>"),
                                album_limit_value,
                            )
                        continue

                    album_counts[album_name] = current_count + 1
                    limited_tracks.append(track)

                if removed_for_album_limit:
                    log.info(
                        "Applied album limit (%s) for playlist '%s' – removed %s track(s)",
                        album_limit_value,
                        name,
                        removed_for_album_limit,
                    )

                matched_tracks = limited_tracks
                match_count = len(matched_tracks)
            else:
                log.warning(
                    "Album limit for playlist '%s' must be positive; received %s.",
                    name,
                    album_limit_raw,
                )

    if limit:
        matched_tracks = matched_tracks[:limit]
        match_count = len(matched_tracks)

    after_sort_duration = _sort_tracks_in_place(
        matched_tracks,
        resolved_after_sort,
        after_sort_desc,
        log,
        dedup_popularity_cache,
        album_popularity_cache,
        album_popularity_cache_by_object,
        debug_logging=debug_logging,
    )
    sort_duration += after_sort_duration

    log.info(f"Playlist '{name}' → {match_count} matching tracks")

    if existing and not deleted_existing:
        try:
            existing.delete()
        except Exception as exc:
            log.warning(f"Failed to delete existing playlist '{name}': {exc}")

    if not matched_tracks:
        log.info(f"No tracks matched for '{name}'. Playlist will not be recreated.")
        total_duration = time.perf_counter() - build_start
        log.info(
            "Performance summary for '%s': fetch=%.2fs, prefetch=%.2fs, filter=%.2fs (%.1f track/s), dedup=%.2fs, sort=%.2fs, update=%.2fs, total=%.2fs",
            name,
            fetch_duration,
            warm_duration,
            filter_duration,
            filter_rate,
            dedup_duration,
            sort_duration,
            playlist_update_duration,
            total_duration,
        )
        log.info(f"✅ Finished building '{name}' (0 tracks)")
        return

    update_start = time.perf_counter()
    for chunk in chunked(matched_tracks, chunk_size):
        try:
            if playlist_obj is None:
                playlist_obj = plex_server.createPlaylist(name, items=chunk)
            else:
                playlist_obj.addItems(chunk)
        except Exception as exc:
            log.error(f"Failed to update playlist '{name}': {exc}")
            raise
    playlist_update_duration += time.perf_counter() - update_start

    apply_playlist_cover(playlist_obj, cover_path)
    total_duration = time.perf_counter() - build_start
    update_rate = (match_count / playlist_update_duration) if playlist_update_duration > 0 else 0.0
    log.info(
        "Performance summary for '%s': fetch=%.2fs, prefetch=%.2fs, filter=%.2fs (%.1f track/s), dedup=%.2fs, sort=%.2fs, update=%.2fs (%.1f track/s), total=%.2fs",
        name,
        fetch_duration,
        warm_duration,
        filter_duration,
        filter_rate,
        dedup_duration,
        sort_duration,
        playlist_update_duration,
        update_rate,
        total_duration,
    )
    log.info(f"✅ Finished building '{name}' ({match_count} tracks)")


def process_playlist(name, config):
    playlist_handler = None
    playlist_log_path = None

    try:
        try:
            playlist_handler, playlist_log_path = _create_playlist_log_handler(name)
        except Exception as exc:
            logger.warning(f"Unable to create debug log for '{name}': {exc}")
            playlist_handler = None
            playlist_log_path = None

        log = PlaylistLoggerProxy(logger, playlist_handler)

        previous_thread_logger = getattr(_thread_local_logger, "current", None)
        _thread_local_logger.current = log

        try:
            _run_playlist_build(name, config, log, playlist_handler, playlist_log_path)
        finally:
            if previous_thread_logger is None:
                if hasattr(_thread_local_logger, "current"):
                    delattr(_thread_local_logger, "current")
            else:
                _thread_local_logger.current = previous_thread_logger
    finally:
        flush_metadata_cache()
        AllMusicPopularityProvider.save_shared_cache()
        if playlist_handler:
            log.debug(
                "Closing per-playlist debug logging for '%s'",
                name,
            )
            playlist_handler.close()

# Main Runtime Logic
# ----------------------------
def _run_playlists(playlists_subset, completion_message=""):
    if not playlists_subset:
        logger.warning("No playlists defined. Nothing to process.")
        return

    playlist_names = list(playlists_subset.keys())
    logger.info(
        "Processing %d playlist(s): %s",
        len(playlist_names),
        ", ".join(playlist_names),
    )

    errors = False
    configured_workers = MAX_WORKERS
    if not isinstance(configured_workers, int) or configured_workers < 1:
        logger.warning(
            "Invalid runtime.max_workers value '%s'; defaulting to 1 worker.",
            configured_workers,
        )
        configured_workers = 1

    worker_count = max(1, min(configured_workers, len(playlists_subset)))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(process_playlist, name, cfg): name
            for name, cfg in playlists_subset.items()
        }
        for future in as_completed(futures):
            playlist_name = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors = True
                logger.exception(f"Playlist '{playlist_name}' failed: {exc}")

    if errors:
        raise RuntimeError("One or more playlists failed to build.")

    if completion_message:
        logger.info(completion_message)


def run_all_playlists():
    _run_playlists(playlists_data, "✅ All playlists processed successfully.")


def run_selected_playlists(playlist_names):
    if not playlist_names:
        logger.warning("No playlists provided. Nothing to process.")
        return

    missing = [name for name in playlist_names if name not in playlists_data]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Unknown playlist(s): {missing_list}")

    selected = {name: playlists_data[name] for name in playlist_names}
    _run_playlists(selected, "✅ Selected playlists processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plex playlist builder")
    parser.add_argument(
        "--playlist",
        dest="playlists",
        action="append",
        help="Name of a playlist to build. Can be provided multiple times.",
    )
    args = parser.parse_args()

    if CACHE_ONLY:
        build_metadata_cache()
        logger.info("Cache-only mode complete. Exiting.")
    elif args.playlists:
        try:
            run_selected_playlists(args.playlists)
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(1)
        except Exception as exc:
            logger.exception(f"Error while processing playlists: {exc}")
            sys.exit(1)
    elif RUN_FOREVER:
        logger.info("Running in loop mode.")
        while True:
            try:
                run_all_playlists()
            except Exception as exc:
                logger.exception(f"Error while processing playlists: {exc}")
            logger.info(f"Sleeping for {REFRESH_INTERVAL} minutes before next run...")
            try:
                time.sleep(max(REFRESH_INTERVAL, 0) * 60)
            except KeyboardInterrupt:
                logger.info("Loop interrupted by user. Exiting.")
                break
    else:
        run_all_playlists()
