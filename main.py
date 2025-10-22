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
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cmp_to_key
from xml.etree import ElementTree as ET
from html import unescape
import unicodedata
from urllib.parse import unquote, quote

from plexapi.server import PlexServer
from spotipy import Spotify
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials
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


spotify_cfg = cfg.get("spotify", {}) or {}
SPOTIFY_CLIENT_ID = spotify_cfg.get("client_id")
SPOTIFY_CLIENT_SECRET = spotify_cfg.get("client_secret")
SPOTIFY_MARKET = spotify_cfg.get("market")
SPOTIFY_SEARCH_LIMIT = spotify_cfg.get("search_limit", 10)

_spotify_min_interval = _coerce_positive_float(
    spotify_cfg.get("request_min_interval_seconds")
)
if _spotify_min_interval is None:
    SPOTIFY_REQUEST_MIN_INTERVAL = 0.25
else:
    SPOTIFY_REQUEST_MIN_INTERVAL = _spotify_min_interval

_spotify_retry_backoff = _coerce_positive_float(
    spotify_cfg.get("request_retry_backoff_seconds")
)
if _spotify_retry_backoff is None:
    SPOTIFY_REQUEST_RETRY_BACKOFF = 5.0
else:
    SPOTIFY_REQUEST_RETRY_BACKOFF = _spotify_retry_backoff


_spotify_chunk_setting = spotify_cfg.get("population_chunk_size")
if isinstance(_spotify_chunk_setting, bool) and not _spotify_chunk_setting:
    SPOTIFY_POPULATION_CHUNK_SIZE = None
else:
    _spotify_chunk_size = _coerce_positive_int(_spotify_chunk_setting)
    if _spotify_chunk_size is None:
        SPOTIFY_POPULATION_CHUNK_SIZE = 10
    else:
        SPOTIFY_POPULATION_CHUNK_SIZE = _spotify_chunk_size

_spotify_chunk_delay_setting = spotify_cfg.get("population_chunk_delay_seconds")
if isinstance(_spotify_chunk_delay_setting, bool) and not _spotify_chunk_delay_setting:
    SPOTIFY_POPULATION_CHUNK_DELAY = 0.0
else:
    _spotify_chunk_delay = _coerce_positive_float(_spotify_chunk_delay_setting)
    if _spotify_chunk_delay is None:
        SPOTIFY_POPULATION_CHUNK_DELAY = SPOTIFY_REQUEST_RETRY_BACKOFF
    else:
        SPOTIFY_POPULATION_CHUNK_DELAY = _spotify_chunk_delay


_spotify_cache_file_setting = spotify_cfg.get("cache_file")
if isinstance(_spotify_cache_file_setting, bool) and not _spotify_cache_file_setting:
    SPOTIFY_CACHE_FILE = None
else:
    SPOTIFY_CACHE_FILE = _resolve_path_setting(
        _spotify_cache_file_setting,
        RUNTIME_DIR / "spotify_popularity.json",
        CONFIG_DIR,
    )

_spotify_state_file_setting = spotify_cfg.get("popularity_state_file")
if isinstance(_spotify_state_file_setting, bool) and not _spotify_state_file_setting:
    SPOTIFY_POPULARITY_STATE_FILE = None
else:
    SPOTIFY_POPULARITY_STATE_FILE = _resolve_path_setting(
        _spotify_state_file_setting,
        RUNTIME_DIR / "spotify_popularity_state.json",
        CONFIG_DIR,
    )

_spotify_cache_max_age_days = _coerce_positive_float(spotify_cfg.get("cache_max_age_days"))
if _spotify_cache_max_age_days is not None:
    SPOTIFY_CACHE_MAX_AGE_SECONDS = _spotify_cache_max_age_days * 86400.0
else:
    SPOTIFY_CACHE_MAX_AGE_SECONDS = None

SPOTIFY_CACHE_VERSION = 1

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
ACTIVE_SPOTIFY_LOG_FILE = None

_default_log_dir = os.path.dirname(LOG_FILE) if LOG_FILE else ""
if not _default_log_dir:
    _default_log_dir = str((RUNTIME_DIR / "logs").resolve())

_spotify_activity_setting = logging_cfg.get("spotify_activity_file")
if isinstance(_spotify_activity_setting, bool) and not _spotify_activity_setting:
    SPOTIFY_ACTIVITY_LOG_FILE = None
else:
    default_spotify_log_path = (Path(_default_log_dir) / "spotify_activity.log").resolve()
    SPOTIFY_ACTIVITY_LOG_FILE = _resolve_path_setting(
        _spotify_activity_setting,
        default_spotify_log_path,
        CONFIG_DIR,
    )

PLAYLIST_LOG_DIR = logging_cfg.get("playlist_debug_dir")
if isinstance(PLAYLIST_LOG_DIR, str) and not PLAYLIST_LOG_DIR.strip():
    PLAYLIST_LOG_DIR = None
if PLAYLIST_LOG_DIR is None:
    PLAYLIST_LOG_DIR = os.path.join(_default_log_dir, "playlists")

if not PLEX_URL or not PLEX_TOKEN:
    raise EnvironmentError("PLEX_URL and PLEX_TOKEN must be set in config.yml")


class SpotifyActivityFilter(logging.Filter):
    """Filter log records so only Spotify-tagged entries are captured."""

    def filter(self, record):
        return bool(getattr(record, "spotify_activity", False))


def setup_logging():
    """Configure logging to stream to stdout and a persistent log file."""
    logger_name = "plex_music_builder"
    logger_obj = logging.getLogger(logger_name)

    # Avoid duplicating handlers if setup_logging is called multiple times
    if logger_obj.handlers:
        return logger_obj

    global ACTIVE_LOG_FILE, ACTIVE_SPOTIFY_LOG_FILE

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

    spotify_logger_obj = logging.getLogger(f"{logger_name}.spotify")
    spotify_logger_obj.setLevel(logging.DEBUG)
    spotify_logger_obj.propagate = True

    if SPOTIFY_ACTIVITY_LOG_FILE:
        spotify_log_dir = os.path.dirname(SPOTIFY_ACTIVITY_LOG_FILE)
        if spotify_log_dir and not os.path.exists(spotify_log_dir):
            os.makedirs(spotify_log_dir, exist_ok=True)

        truncate_error = False
        try:
            with open(SPOTIFY_ACTIVITY_LOG_FILE, "w", encoding="utf-8"):
                pass
        except OSError as exc:
            truncate_error = True
            logger_obj.error(
                f"Unable to initialise Spotify log file '{SPOTIFY_ACTIVITY_LOG_FILE}' for writing: {exc}"
            )

        if not truncate_error:
            try:
                spotify_handler = TimedRotatingFileHandler(
                    SPOTIFY_ACTIVITY_LOG_FILE,
                    when="midnight",
                    backupCount=7,
                    encoding="utf-8",
                    utc=False,
                    delay=False,
                    interval=1,
                )
            except OSError as exc:
                logger_obj.error(
                    f"Unable to open Spotify log file '{SPOTIFY_ACTIVITY_LOG_FILE}': {exc}"
                )
            else:
                spotify_handler.setFormatter(formatter)
                spotify_handler.setLevel(logging.DEBUG)
                spotify_handler.addFilter(SpotifyActivityFilter())
                spotify_logger_obj.addHandler(spotify_handler)
                ACTIVE_SPOTIFY_LOG_FILE = SPOTIFY_ACTIVITY_LOG_FILE

    return logger_obj


logger = setup_logging()
spotify_logger = logging.LoggerAdapter(
    logging.getLogger(f"{logger.name}.spotify"), {"spotify_activity": True}
)
if ACTIVE_LOG_FILE:
    logger.info(f"Detailed logs will be written to: {ACTIVE_LOG_FILE}")
if ACTIVE_SPOTIFY_LOG_FILE:
    spotify_logger.info(
        f"Spotify activity logs will be written to: {ACTIVE_SPOTIFY_LOG_FILE}"
    )


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
plex = PlexServer(PLEX_URL, PLEX_TOKEN)

# ----------------------------
# Load Playlists Definition
# ----------------------------
raw_playlists = load_yaml(PLAYLISTS_FILE)


def _merge_playlist_defaults(raw_payload):
    if not isinstance(raw_payload, dict):
        return {}

    defaults = raw_payload.get("defaults")
    playlists = raw_payload.get("playlists")

    if not isinstance(playlists, dict):
        return {}

    default_cfg = copy.deepcopy(defaults) if isinstance(defaults, dict) else {}
    default_filters = default_cfg.pop("plex_filter", []) or []

    merged_playlists = {}
    for playlist_name, playlist_cfg in playlists.items():
        cfg = copy.deepcopy(playlist_cfg) if isinstance(playlist_cfg, dict) else {}

        combined = copy.deepcopy(default_cfg)
        combined.update(cfg)

        playlist_filters = cfg.get("plex_filter", []) or []
        combined_filters = []
        if default_filters:
            combined_filters.extend(copy.deepcopy(default_filters))
        if playlist_filters:
            combined_filters.extend(copy.deepcopy(playlist_filters))

        if default_filters or playlist_filters or "plex_filter" in combined:
            combined["plex_filter"] = combined_filters

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

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_cache, f)
def _strip_parenthetical(value):
    if not value:
        return ""
    return re.sub(r"\s*\([^)]*\)\s*", " ", str(value)).strip()


def _normalize_compare_value(value):
    if not value:
        return ""
    normalized = re.sub(r"\s+", " ", str(value)).strip().lower()
    return normalized


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


def _compute_spotify_popularity_score(track, spotify_provider=None, playlist_logger=None):
    if not (spotify_provider and getattr(spotify_provider, "is_enabled", False)):
        return None

    profile = spotify_provider.get_track_profile(track)
    if profile is None:
        return None

    if isinstance(profile, dict):
        popularity = _coerce_non_negative_float(profile.get("popularity"))
    else:
        popularity = _coerce_non_negative_float(profile)

    if popularity is None:
        return None

    score = popularity

    metrics_container = profile if isinstance(profile, dict) else {}

    play_count = None
    like_count = None

    candidates = []
    if isinstance(metrics_container, dict):
        candidates.append(metrics_container)
        for nested_key in ("metrics", "statistics", "insights"):
            nested = metrics_container.get(nested_key)
            if isinstance(nested, dict):
                candidates.append(nested)
        album_metrics = metrics_container.get("album") if isinstance(metrics_container.get("album"), dict) else None
        if isinstance(album_metrics, dict):
            candidates.append(album_metrics)

    for container in candidates:
        if play_count is None:
            play_count = _extract_numeric_from_candidates(
                container,
                ("play_count", "plays", "playcount", "streams"),
            )
        if like_count is None:
            like_count = _extract_numeric_from_candidates(
                container,
                ("like_count", "likes", "favourites", "favorites", "hearts"),
            )

    if play_count is not None:
        play_bonus = min(10.0, math.log1p(play_count))
        score += play_bonus

    if like_count is not None:
        like_bonus = min(10.0, math.log1p(like_count))
        score += like_bonus

    return round(score, 3)


def _resolve_track_popularity_value(track, spotify_provider=None, playlist_logger=None):
    return _compute_spotify_popularity_score(
        track,
        spotify_provider=spotify_provider,
        playlist_logger=playlist_logger,
    )


def _deduplicate_tracks(tracks, log, spotify_provider=None):
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
            spotify_provider=spotify_provider,
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
    spotify_provider=None,
    playlist_logger=None,
):
    if not tracks:
        return {}, {}

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
                spotify_provider=spotify_provider,
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
            adjusted_score = base_score * 1.5 if index < 5 else base_score
            if cache_key_str:
                adjusted_by_rating_key[cache_key_str] = adjusted_score
            else:
                adjusted_by_object[id(track)] = adjusted_score

    return adjusted_by_rating_key, adjusted_by_object


def _resolve_album_year(track):
    """Return the best-known release year for the provided track's album."""

    def _normalize_year(value):
        token = _extract_year_token(value)
        return token

    year_candidates = [
        getattr(track, "parentYear", None),
        getattr(track, "year", None),
        getattr(track, "originallyAvailableAt", None),
        getattr(track, "parentOriginallyAvailableAt", None),
    ]

    for candidate in year_candidates:
        normalized = _normalize_year(candidate)
        if normalized:
            return normalized

    if CACHE_ONLY:
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
                    return normalized

    return None


def _get_album_guid(track):
    """Return the album GUID (``plex://album/...``) for a track, if available."""

    parent_guid = getattr(track, "parentGuid", None)
    if parent_guid:
        return str(parent_guid)

    # Fall back to the track metadata and, if necessary, the album metadata.
    try:
        track_xml = fetch_full_metadata(track.ratingKey)
        parent_guid = parse_field_from_xml(track_xml, "parentGuid")
        if parent_guid:
            return str(parent_guid)
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
            return str(candidate_guid)

    return None


def _get_track_guid(track):
    """Return the track GUID (``plex://track/...``) for a track, if available."""

    track_guid = getattr(track, "guid", None)
    if track_guid:
        return str(track_guid)

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
    return str(guid) if guid else None


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


class SpotifyPopularityProvider:
    """Shared Spotify client that resolves track popularity for sorting."""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._log = spotify_logger
        self._client = None
        self._client_lock = threading.Lock()
        self._track_cache = {}
        self._id_cache = {}
        self._query_cache = {}
        self._enabled = False
        self._error = None
        self._cache_file = SPOTIFY_CACHE_FILE
        self._cache_ttl = SPOTIFY_CACHE_MAX_AGE_SECONDS
        self._persistent_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_dirty = False
        self._save_counter = 0
        self._state_file = SPOTIFY_POPULARITY_STATE_FILE
        self._last_populated_at = None
        self._last_populated_summary = None
        self.market = self._normalize_market(SPOTIFY_MARKET)
        self.search_limit = self._coerce_search_limit(SPOTIFY_SEARCH_LIMIT)
        self._rate_lock = threading.Lock()
        self._next_allowed_request = 0.0
        self._backoff_until = 0.0
        self._rate_limit_resume_wall = 0.0
        self._min_request_interval = max(0.0, SPOTIFY_REQUEST_MIN_INTERVAL or 0.0)
        self._retry_backoff_seconds = max(0.0, SPOTIFY_REQUEST_RETRY_BACKOFF or 0.0)
        self._population_state_lock = threading.Lock()
        self._population_state = None

        self._load_cache()
        self._load_state()

        if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
            self._error = "Missing Spotify client credentials."
            return

        try:
            auth_manager = SpotifyClientCredentials(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
            )
            self._client = Spotify(
                auth_manager=auth_manager,
                requests_timeout=10,
                retries=3,
            )
            self._enabled = True
        except Exception as exc:
            self._log.warning("Failed to initialize Spotify client: %s", exc)
            self._error = str(exc)
            self._client = None

    @staticmethod
    def _normalize_market(market):
        if not market:
            return None
        normalized = str(market).strip().upper()
        return normalized or None

    @staticmethod
    def _coerce_search_limit(value):
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = 10
        return max(1, numeric)

    def _throttle_spotify_request(self):
        min_interval = self._min_request_interval

        if min_interval <= 0 and self._backoff_until <= 0:
            return

        should_persist = False
        acquired = False
        while True:
            with self._rate_lock:
                now = time.monotonic()
                wait_until = max(self._next_allowed_request, self._backoff_until)
                if now >= wait_until:
                    if self._backoff_until > 0 and now >= self._backoff_until:
                        self._backoff_until = 0.0
                        if self._rate_limit_resume_wall > 0:
                            self._rate_limit_resume_wall = 0.0
                            should_persist = True
                    next_time = now + min_interval if min_interval > 0 else now
                    self._next_allowed_request = next_time
                    acquired = True
                    break

                sleep_for = wait_until - now

            if sleep_for <= 0:
                break

            time.sleep(min(sleep_for, 5.0))

        if acquired and should_persist:
            self._write_state_file()
        return

    def _register_rate_limit_backoff(self, retry_after=None, minimum=None):
        delay = 0.0

        for candidate in (retry_after, minimum, self._retry_backoff_seconds):
            if candidate is None:
                continue
            try:
                delay = max(delay, float(candidate))
            except (TypeError, ValueError):
                continue

        if delay <= 0:
            return

        monotonic_now = time.monotonic()
        wall_now = time.time()
        resume_at = monotonic_now + delay
        resume_wall = wall_now + delay

        should_persist = False
        with self._rate_lock:
            if resume_at > self._backoff_until:
                self._backoff_until = resume_at
                should_persist = True
            if resume_wall > self._rate_limit_resume_wall:
                self._rate_limit_resume_wall = resume_wall
                should_persist = True

            resume_for_next = resume_at
            if self._min_request_interval > 0:
                resume_for_next += self._min_request_interval
            if resume_for_next > self._next_allowed_request:
                self._next_allowed_request = resume_for_next

        if should_persist:
            self._write_state_file()

    @staticmethod
    def _extract_retry_after(exc):
        if exc is None:
            return None

        def _normalize_retry_value(value, allow_milliseconds=False):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None

            # Some Spotify error responses report retry windows in milliseconds.
            # Treat very large values as milliseconds and convert to seconds.
            if allow_milliseconds and numeric > 1000:
                numeric /= 1000.0

            return numeric if numeric > 0 else None

        header_value = None

        for attr_name in ("headers", "http_headers"):
            header_dict = getattr(exc, attr_name, None)
            if isinstance(header_dict, dict):
                header_value = header_dict.get("Retry-After") or header_dict.get("retry-after")
                if header_value is not None:
                    break

        if header_value is None:
            header_value = getattr(exc, "retry_after", None)

        message = str(exc) if exc is not None else ""
        message_retry = None
        if message:
            match = re.search(r"Retry will occur after:\s*(?P<value>[0-9]+(?:\.[0-9]+)?)", message)
            if match:
                message_retry = _normalize_retry_value(
                    match.group("value"), allow_milliseconds=True
                )

        retry_after = _normalize_retry_value(header_value, allow_milliseconds=False)
        if retry_after is None:
            return message_retry

        if message_retry is not None and retry_after > message_retry * 10:
            return message_retry

        return retry_after

    def get_population_resume_state(self):
        with self._population_state_lock:
            if not self._population_state:
                return None
            return copy.deepcopy(self._population_state)

    def begin_population_run(self, pending_keys, total_tracks, cached_profiles=0):
        if not isinstance(pending_keys, (list, tuple, set)):
            pending_iterable = list(pending_keys or [])
        else:
            pending_iterable = list(pending_keys)

        normalized_keys = [
            str(key)
            for key in pending_iterable
            if key not in (None, "")
        ]

        try:
            total_tracks_value = int(total_tracks) if total_tracks is not None else None
        except (TypeError, ValueError):
            total_tracks_value = None

        try:
            cached_profiles_value = int(cached_profiles)
        except (TypeError, ValueError):
            cached_profiles_value = 0

        state = {
            "pending_keys": normalized_keys,
            "total_tracks": total_tracks_value,
            "processed": 0,
            "new_profiles": 0,
            "refreshed_profiles": 0,
            "cached_profiles": cached_profiles_value,
            "missing_profiles": 0,
            "errors": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        with self._population_state_lock:
            self._population_state = state

        self._write_state_file()
        return copy.deepcopy(state)

    def update_population_resume(self, rating_key, outcome=None):
        rating_key_str = None if rating_key in (None, "") else str(rating_key)

        with self._population_state_lock:
            if not self._population_state:
                return

            state = self._population_state
            pending = state.get("pending_keys", [])

            if rating_key_str is not None and isinstance(pending, list):
                if pending and pending[0] == rating_key_str:
                    pending.pop(0)
                else:
                    try:
                        pending.remove(rating_key_str)
                    except ValueError:
                        pass

            state["processed"] = state.get("processed", 0) + 1

            if outcome == "new":
                state["new_profiles"] = state.get("new_profiles", 0) + 1
            elif outcome == "refreshed":
                state["refreshed_profiles"] = state.get("refreshed_profiles", 0) + 1
            elif outcome == "cached":
                state["cached_profiles"] = state.get("cached_profiles", 0) + 1
            elif outcome == "missing":
                state["missing_profiles"] = state.get("missing_profiles", 0) + 1
            elif outcome == "error":
                state["errors"] = state.get("errors", 0) + 1

        self._write_state_file()

    def clear_population_resume(self):
        with self._population_state_lock:
            self._population_state = None

        self._write_state_file()

    def _load_cache(self):
        if not self._cache_file:
            return
        if not os.path.exists(self._cache_file):
            return

        try:
            with open(self._cache_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            self._log.warning(
                "Unable to load Spotify cache from '%s': %s",
                self._cache_file,
                exc,
            )
            return

        if not isinstance(payload, dict):
            return

        if payload.get("version") != SPOTIFY_CACHE_VERSION:
            self._log.debug(
                "Ignoring Spotify cache with mismatched version (%s)",
                payload.get("version"),
            )
            return

        tracks = payload.get("tracks", {})
        if not isinstance(tracks, dict):
            return

        for spotify_id, entry in tracks.items():
            if not spotify_id:
                continue
            if isinstance(entry, dict):
                profile = entry.get("profile")
                fetched_at = entry.get("fetched_at")
            else:
                profile = entry
                fetched_at = None

            self._persistent_cache[str(spotify_id)] = {
                "profile": profile,
                "fetched_at": fetched_at,
            }

        last_populated = payload.get("last_populated_at")
        if last_populated:
            self._last_populated_at = last_populated

        summary = payload.get("last_populated_summary")
        if isinstance(summary, dict):
            self._last_populated_summary = summary

    def _load_state(self):
        if not self._state_file:
            return
        if not os.path.exists(self._state_file):
            return

        try:
            with open(self._state_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            self._log.debug(
                "Unable to load Spotify popularity state file '%s': %s",
                self._state_file,
                exc,
            )
            return

        if not isinstance(payload, dict):
            return

        last_populated = payload.get("last_populated_at")
        summary = payload.get("summary")
        resume = payload.get("in_progress")
        rate_limit_payload = payload.get("rate_limit")

        if last_populated:
            self._last_populated_at = last_populated
        if isinstance(summary, dict):
            self._last_populated_summary = summary

        if isinstance(resume, dict):
            pending = resume.get("pending_keys")
            if isinstance(pending, list):
                normalized_pending = [
                    str(key)
                    for key in pending
                    if key not in (None, "")
                ]
            else:
                normalized_pending = []

            def _coerce_int(value, default=0):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return default

            total_tracks = resume.get("total_tracks")
            try:
                total_tracks = int(total_tracks) if total_tracks is not None else None
            except (TypeError, ValueError):
                total_tracks = None

            normalized_state = {
                "pending_keys": normalized_pending,
                "total_tracks": total_tracks,
                "processed": _coerce_int(resume.get("processed"), 0),
                "new_profiles": _coerce_int(resume.get("new_profiles"), 0),
                "refreshed_profiles": _coerce_int(resume.get("refreshed_profiles"), 0),
                "cached_profiles": _coerce_int(resume.get("cached_profiles"), 0),
                "missing_profiles": _coerce_int(resume.get("missing_profiles"), 0),
                "errors": _coerce_int(resume.get("errors"), 0),
                "started_at": resume.get("started_at"),
            }

            with self._population_state_lock:
                self._population_state = normalized_state

        resume_epoch = None
        if isinstance(rate_limit_payload, dict):
            resume_epoch = rate_limit_payload.get("resume_epoch")
            if resume_epoch is None:
                resume_epoch = rate_limit_payload.get("resume_at")
        elif rate_limit_payload is not None:
            resume_epoch = rate_limit_payload

        try:
            resume_epoch = float(resume_epoch) if resume_epoch is not None else None
        except (TypeError, ValueError):
            resume_epoch = None

        if resume_epoch:
            wall_now = time.time()
            remaining = resume_epoch - wall_now
            if remaining > 0:
                resume_at = time.monotonic() + remaining
                with self._rate_lock:
                    if resume_at > self._backoff_until:
                        self._backoff_until = resume_at
                    if resume_epoch > self._rate_limit_resume_wall:
                        self._rate_limit_resume_wall = resume_epoch
                    next_allowed = resume_at
                    if self._min_request_interval > 0:
                        next_allowed += self._min_request_interval
                    if next_allowed > self._next_allowed_request:
                        self._next_allowed_request = next_allowed

    def _remember_profile(self, spotify_id, profile):
        if not spotify_id:
            return

        spotify_id = str(spotify_id)
        self._id_cache[spotify_id] = profile

        entry = {
            "profile": profile,
            "fetched_at": time.time(),
        }

        with self._cache_lock:
            self._persistent_cache[spotify_id] = entry
            if self._cache_file:
                self._cache_dirty = True
                self._save_counter += 1
                if SAVE_INTERVAL and self._save_counter >= SAVE_INTERVAL:
                    self.save_cache()

    def _get_profile_from_cache(self, spotify_id, allow_stale=False):
        if not spotify_id:
            return None, False, False

        spotify_id = str(spotify_id)

        if spotify_id in self._id_cache and spotify_id not in self._persistent_cache:
            return self._id_cache[spotify_id], True, False

        entry = self._persistent_cache.get(spotify_id)
        if entry is None:
            return None, False, False

        profile = entry.get("profile") if isinstance(entry, dict) else entry
        fetched_at = entry.get("fetched_at") if isinstance(entry, dict) else None

        is_stale = False
        if self._cache_ttl and fetched_at:
            try:
                age = time.time() - float(fetched_at)
            except (TypeError, ValueError):
                age = 0.0
            if age > self._cache_ttl:
                is_stale = True
                if not allow_stale:
                    return None, False, True

        self._id_cache[spotify_id] = profile
        return profile, True, is_stale

    def _fetch_profile_from_api(self, spotify_id, fallback_profile=None, fallback_found=False):
        if not spotify_id:
            return fallback_profile if fallback_found else None

        spotify_id = str(spotify_id)

        if not self._client:
            return fallback_profile if fallback_found else None

        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            attempts += 1

            self._throttle_spotify_request()

            try:
                with self._client_lock:
                    data = self._client.track(spotify_id)
            except SpotifyException as exc:
                status = getattr(exc, "http_status", None)

                if status == 404:
                    self._remember_profile(spotify_id, None)
                    return None

                retry_after = self._extract_retry_after(exc)

                if status == 429:
                    backoff_seconds = retry_after or self._retry_backoff_seconds or 1.0
                    self._log.warning(
                        "Spotify rate limit encountered for %s; backing off for %.2fs",
                        spotify_id,
                        backoff_seconds,
                    )
                    self._register_rate_limit_backoff(retry_after)
                    if attempts < max_attempts:
                        continue
                elif status in {500, 502, 503, 504}:
                    self._log.debug(
                        "Transient Spotify error (%s) during track lookup for %s; retrying",
                        status,
                        spotify_id,
                    )
                    self._register_rate_limit_backoff(
                        max(
                            retry_after or 0.0,
                            (self._retry_backoff_seconds or 1.0) * attempts,
                        )
                    )
                    if attempts < max_attempts:
                        continue
                else:
                    self._log.debug(
                        "Spotify track lookup failed for %s: %s",
                        spotify_id,
                        exc,
                    )
                    if retry_after:
                        self._register_rate_limit_backoff(retry_after)

                if fallback_found:
                    return fallback_profile
                return None
            except Exception as exc:
                self._log.debug(
                    "Unexpected Spotify error during track lookup for %s: %s",
                    spotify_id,
                    exc,
                )
                self._register_rate_limit_backoff(self._retry_backoff_seconds)
                if attempts < max_attempts:
                    continue
                if fallback_found:
                    return fallback_profile
                return None

            profile = self._extract_track_profile(data)
            self._remember_profile(spotify_id, profile)
            return profile

        return fallback_profile if fallback_found else None

    def _resolve_profile_by_id(self, spotify_id):
        profile, found, is_stale = self._get_profile_from_cache(spotify_id)
        if found:
            if profile is None:
                return None, "missing_cache" if not is_stale else "stale"
            return profile, "hit"

        fallback_profile, fallback_found, fallback_stale = self._get_profile_from_cache(
            spotify_id,
            allow_stale=True,
        )

        profile = self._fetch_profile_from_api(
            spotify_id,
            fallback_profile,
            fallback_found,
        )

        if profile is not None:
            return profile, "hit"

        if fallback_found:
            if fallback_profile is None:
                return None, "missing_cache" if not fallback_stale else "stale"
            return fallback_profile, "hit"

        return None, "miss"

    def save_cache(self):
        should_persist_cache = bool(self._cache_file and self._cache_dirty)
        should_write_state = bool(self._state_file and self._last_populated_at)

        if not should_persist_cache and not should_write_state:
            return

        payload = None
        if should_persist_cache:
            cache_dir = os.path.dirname(self._cache_file)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)

            with self._cache_lock:
                tracks_payload = {}
                for spotify_id, entry in self._persistent_cache.items():
                    if isinstance(entry, dict):
                        tracks_payload[spotify_id] = {
                            "profile": entry.get("profile"),
                            "fetched_at": entry.get("fetched_at"),
                        }
                    else:
                        tracks_payload[spotify_id] = {
                            "profile": entry,
                            "fetched_at": None,
                        }

                payload = {
                    "version": SPOTIFY_CACHE_VERSION,
                    "tracks": tracks_payload,
                }
                if self._last_populated_at:
                    payload["last_populated_at"] = self._last_populated_at
                if isinstance(self._last_populated_summary, dict):
                    payload["last_populated_summary"] = self._last_populated_summary

            try:
                with open(self._cache_file, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2)
            except Exception as exc:
                self._log.warning(
                    "Unable to persist Spotify cache to '%s': %s",
                    self._cache_file,
                    exc,
                )
            else:
                self._cache_dirty = False
                self._save_counter = 0

        if should_write_state:
            self._write_state_file()

    def _write_state_file(self):
        if not self._state_file:
            return

        with self._cache_lock:
            last_populated = self._last_populated_at
            summary = copy.deepcopy(self._last_populated_summary)
            if not isinstance(summary, dict):
                summary = {}

        with self._population_state_lock:
            in_progress = copy.deepcopy(self._population_state)

        with self._rate_lock:
            resume_epoch = self._rate_limit_resume_wall

        payload = {
            "version": SPOTIFY_CACHE_VERSION,
            "cache_file": self._cache_file,
            "last_populated_at": last_populated,
            "summary": summary or {},
        }

        if in_progress:
            payload["in_progress"] = in_progress

        if resume_epoch and resume_epoch > 0:
            now_wall = time.time()
            if resume_epoch > now_wall:
                payload["rate_limit"] = {"resume_epoch": resume_epoch}

        state_dir = os.path.dirname(self._state_file)
        if state_dir and not os.path.exists(state_dir):
            os.makedirs(state_dir, exist_ok=True)

        try:
            with open(self._state_file, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._log.debug(
                "Unable to persist Spotify popularity state to '%s': %s",
                self._state_file,
                exc,
            )

    def record_population_run(self, summary=None):
        timestamp = datetime.now(timezone.utc).isoformat()
        summary_payload = summary or {}

        with self._cache_lock:
            self._last_populated_at = timestamp
            self._last_populated_summary = summary_payload
            if self._cache_file:
                self._cache_dirty = True

        with self._population_state_lock:
            self._population_state = None

        self._write_state_file()
        return timestamp

    def get_population_state(self):
        with self._cache_lock:
            summary_copy = copy.deepcopy(self._last_populated_summary)
            return self._last_populated_at, summary_copy if isinstance(summary_copy, dict) else None

    def inspect_track_cache(self, track):
        track_key = getattr(track, "ratingKey", None)
        if track_key is not None and track_key in self._track_cache:
            cached_profile = self._track_cache[track_key]
            return cached_profile, True, False

        for spotify_id in self._iter_spotify_track_ids(track):
            profile, found, is_stale = self._get_profile_from_cache(
                spotify_id,
                allow_stale=True,
            )
            if found:
                return profile, True, is_stale

        return None, False, False

    @classmethod
    def save_shared_cache(cls):
        instance = cls._instance
        if instance is not None:
            instance.save_cache()

    @classmethod
    def get_shared(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def is_enabled(self):
        return (self._enabled and self._client is not None) or bool(self._persistent_cache)

    def describe_error(self):
        return self._error

    def get_popularity(self, track):
        return _compute_spotify_popularity_score(track, spotify_provider=self)

    def get_track_profile(self, track):
        if not self.is_enabled:
            return None

        track_key = getattr(track, "ratingKey", None)
        cached_profile = self._track_cache.get(track_key) if track_key else None
        if cached_profile is not None:
            # Backwards compatibility for caches populated before profile support.
            if isinstance(cached_profile, dict):
                return cached_profile
            return {"popularity": cached_profile}

        resolved_profile = None
        saw_missing_cache = False

        for spotify_id in self._iter_spotify_track_ids(track):
            profile_candidate, status = self._resolve_profile_by_id(spotify_id)
            if status == "hit":
                resolved_profile = profile_candidate
                break
            if status == "missing_cache":
                saw_missing_cache = True

        if resolved_profile is None and self._client and not saw_missing_cache:
            resolved_profile = self._search_for_track_profile(track)
            if isinstance(resolved_profile, dict):
                track_id = resolved_profile.get("id")
                if track_id:
                    self._remember_profile(track_id, resolved_profile)

        if track_key is not None:
            self._track_cache[track_key] = resolved_profile

        return resolved_profile

    def _iter_spotify_track_ids(self, track):
        seen = set()
        for guid in self._iter_guid_strings(track):
            spotify_id = self._extract_spotify_track_id(guid)
            if spotify_id and spotify_id not in seen:
                seen.add(spotify_id)
                yield spotify_id

    @staticmethod
    def _iter_guid_strings(track):
        candidates = []
        direct_guid = getattr(track, "guid", None)
        if direct_guid:
            candidates.append(direct_guid)

        for attr_name in ("guids", "providerGuids"):
            attr = getattr(track, attr_name, None)
            if not attr:
                continue
            for item in attr:
                if isinstance(item, str):
                    candidates.append(item)
                else:
                    value = getattr(item, "id", None) or getattr(item, "idUri", None)
                    if value:
                        candidates.append(value)

        for candidate in candidates:
            if candidate:
                yield str(candidate)

    @staticmethod
    def _extract_spotify_track_id(guid):
        if not guid:
            return None
        guid_str = str(guid)
        match = re.search(r"spotify[:/]+track[:/](?P<id>[A-Za-z0-9]+)", guid_str)
        if not match:
            match = re.search(r"spotify:track:(?P<id>[A-Za-z0-9]+)", guid_str)
        if not match:
            return None
        track_id = match.group("id")
        if not track_id:
            return None
        return track_id.split("?")[0]

    def _search_for_track_profile(self, track):
        query = self._build_search_query(track)
        if not query:
            return None

        cache_key = (query, self.market)
        if cache_key in self._query_cache:
            candidates = self._query_cache[cache_key]
        else:
            attempts = 0
            max_attempts = 3

            while attempts < max_attempts:
                attempts += 1

                self._throttle_spotify_request()

                try:
                    with self._client_lock:
                        response = self._client.search(
                            q=query,
                            type="track",
                            market=self.market,
                            limit=self.search_limit,
                        )
                except SpotifyException as exc:
                    status = getattr(exc, "http_status", None)
                    retry_after = self._extract_retry_after(exc)

                    if status == 429:
                        backoff_seconds = retry_after or self._retry_backoff_seconds or 1.0
                        self._log.warning(
                            "Spotify rate limit encountered during search for '%s'; backing off for %.2fs",
                            query,
                            backoff_seconds,
                        )
                        self._register_rate_limit_backoff(retry_after)
                        if attempts < max_attempts:
                            continue
                    elif status in {500, 502, 503, 504}:
                        self._log.debug(
                            "Transient Spotify search error (%s) for '%s'; retrying",
                            status,
                            query,
                        )
                        self._register_rate_limit_backoff(
                            max(
                                retry_after or 0.0,
                                (self._retry_backoff_seconds or 1.0) * attempts,
                            )
                        )
                        if attempts < max_attempts:
                            continue
                    else:
                        self._log.debug("Spotify search failed for '%s': %s", query, exc)
                        if retry_after:
                            self._register_rate_limit_backoff(retry_after)

                    self._query_cache[cache_key] = []
                    return None
                except Exception as exc:
                    self._log.debug("Unexpected Spotify error during search for '%s': %s", query, exc)
                    self._register_rate_limit_backoff(self._retry_backoff_seconds)
                    if attempts < max_attempts:
                        continue
                    self._query_cache[cache_key] = []
                    return None
                else:
                    break
            else:
                self._query_cache[cache_key] = []
                return None

            candidates = response.get("tracks", {}).get("items", []) if isinstance(response, dict) else []
            self._query_cache[cache_key] = candidates

        if not candidates:
            return None

        best_candidate = self._select_best_candidate(track, candidates)
        if not best_candidate and candidates:
            best_candidate = max(
                candidates,
                key=lambda item: (item or {}).get("popularity") or -1,
            )

        if not best_candidate:
            return None
        return self._extract_track_profile(best_candidate)

    def _build_search_query(self, track):
        title = getattr(track, "title", None)
        if not title:
            return None

        stripped_title = _strip_parenthetical(title)
        if stripped_title:
            title = stripped_title

        def sanitize(value):
            if value is None:
                return None
            return str(value).replace('"', "").strip()

        parts = [f'track:"{sanitize(title)}"']

        artist = getattr(track, "grandparentTitle", None) or getattr(track, "originalTitle", None)
        if artist:
            parts.append(f'artist:"{sanitize(artist)}"')

        album = getattr(track, "parentTitle", None)
        if album:
            parts.append(f'album:"{sanitize(album)}"')

        return " ".join(part for part in parts if part)

    def _select_best_candidate(self, track, candidates):
        track_title_norm = _normalize_compare_value(getattr(track, "title", ""))
        track_artist_norms = _collect_normalized_candidates(
            [
                getattr(track, "grandparentTitle", None),
                getattr(track, "originalTitle", None),
            ]
        )
        track_album_norms = _collect_normalized_candidates([getattr(track, "parentTitle", None)])

        best_candidate = None
        best_score = float("-inf")

        for item in candidates:
            if not isinstance(item, dict):
                continue

            popularity = item.get("popularity") or 0
            try:
                popularity_score = float(popularity) / 100.0
            except (TypeError, ValueError):
                popularity_score = 0.0

            item_title_norm = _normalize_compare_value(item.get("name"))
            item_artist_norms = _collect_normalized_candidates(
                artist.get("name") for artist in item.get("artists", []) if isinstance(artist, dict)
            )
            album_info = item.get("album") if isinstance(item.get("album"), dict) else item.get("album")
            album_name = album_info.get("name") if isinstance(album_info, dict) else None
            item_album_norms = _collect_normalized_candidates([album_name])

            score = popularity_score

            if track_title_norm and item_title_norm == track_title_norm:
                score += 5
            elif track_title_norm and item_title_norm and track_title_norm in item_title_norm:
                score += 2

            if track_artist_norms and item_artist_norms & track_artist_norms:
                score += 3

            if track_album_norms and item_album_norms & track_album_norms:
                score += 1

            if score > best_score:
                best_score = score
                best_candidate = item

        return best_candidate

    @staticmethod
    def _extract_track_profile(data):
        if not isinstance(data, dict):
            return None

        popularity = data.get("popularity")
        try:
            popularity = float(popularity) if popularity is not None else None
        except (TypeError, ValueError):
            popularity = None

        track_id = data.get("id")
        uri = data.get("uri")
        name = data.get("name")
        track_number = data.get("track_number")

        album_info = data.get("album") if isinstance(data.get("album"), dict) else {}
        album_type = str(album_info.get("album_type") or album_info.get("type") or "").lower()
        album_group = str(album_info.get("album_group") or "").lower()
        album_name = album_info.get("name")
        release_date = album_info.get("release_date")
        release_date_precision = album_info.get("release_date_precision")

        is_single = False
        if album_type == "single" or album_group == "single":
            is_single = True
        elif isinstance(data.get("single"), bool):
            is_single = data.get("single")

        metrics_candidates = [data]
        statistics_info = data.get("statistics")
        if isinstance(statistics_info, dict):
            metrics_candidates.append(statistics_info)
        insights_info = data.get("insights")
        if isinstance(insights_info, dict):
            metrics_candidates.append(insights_info)
        if isinstance(album_info, dict):
            metrics_candidates.append(album_info)

        play_count = None
        like_count = None
        for candidate in metrics_candidates:
            if play_count is None:
                play_count = _extract_numeric_from_candidates(
                    candidate,
                    ("play_count", "plays", "playcount", "streams"),
                )
            if like_count is None:
                like_count = _extract_numeric_from_candidates(
                    candidate,
                    ("like_count", "likes", "favourites", "favorites", "hearts"),
                )

        metrics = {}
        if play_count is not None:
            metrics["play_count"] = play_count
        if like_count is not None:
            metrics["like_count"] = like_count

        return {
            "id": track_id,
            "uri": uri,
            "name": name,
            "popularity": popularity,
            "album": {
                "name": album_name,
                "album_type": album_type or None,
                "album_group": album_group or None,
                "release_date": release_date,
                "release_date_precision": release_date_precision,
            },
            "is_single": bool(is_single),
            "track_number": track_number,
            "play_count": play_count,
            "like_count": like_count,
            "metrics": metrics,
        }


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
        self._default_spotify_baseline = 50.0
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

    def get_popularity(self, track, spotify_provider=None, playlist_logger=None):
        if not self.is_enabled:
            return None

        title = getattr(track, "title", None)
        artist = getattr(track, "grandparentTitle", None)
        album = getattr(track, "parentTitle", None)

        track_key = self._build_track_key(title, artist, album)
        with self._cache_lock:
            cached_entry = self._track_cache.get(track_key) if track_key else None

        if cached_entry is not None:
            cached_details = cached_entry.get("details") or {}
            if playlist_logger:
                playlist_logger.debug(
                    "AllMusic cache hit for '%s' by '%s' → details=%s",
                    title or "<unknown>",
                    artist or "<unknown>",
                    {k: v for k, v in cached_details.items() if k != "components"},
                )
            details = dict(cached_details)
        else:
            details = None

        if details is None:
            query = self._build_query(title, artist, album)
            if not query:
                if playlist_logger:
                    playlist_logger.debug(
                        "Skipping AllMusic popularity lookup for '%s' – insufficient metadata.",
                        title or "<unknown>",
                    )
                if track_key:
                    with self._cache_lock:
                        self._track_cache[track_key] = {
                            "popularity": None,
                            "timestamp": time.time(),
                            "details": {},
                        }
                        self._cache_dirty = True
                return None

            if playlist_logger:
                playlist_logger.debug(
                    "Fetching AllMusic popularity for '%s' by '%s' (query='%s')",
                    title or "<unknown>",
                    artist or "<unknown>",
                    query,
                )

            with self._cache_lock:
                cached_query = self._query_cache.get(query)

            if cached_query is not None:
                details = dict(cached_query)
                if playlist_logger:
                    playlist_logger.debug(
                        "AllMusic query cache hit for '%s': matched=%s",
                        query,
                        details.get("title"),
                    )
            else:
                details = self._execute_search(query, title, artist, album, playlist_logger)
                if details is None:
                    details = {}

        base_score, computed_details = self._compute_composite_popularity(
            track,
            details,
            spotify_provider=spotify_provider,
            playlist_logger=playlist_logger,
        )

        final_score, finalized_details = self._apply_single_boost(
            track,
            track_key,
            base_score,
            computed_details,
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


    def _compute_composite_popularity(self, track, details, spotify_provider=None, playlist_logger=None):
        details = details or {}
        album_rating_count = details.get("album_rating_count")
        rating_count = details.get("rating_count")

        if album_rating_count is None:
            album_rating_count = rating_count

        normalized_rating = None
        if album_rating_count is not None:
            try:
                numeric = float(album_rating_count)
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None and numeric > 0:
                normalized_rating = math.log1p(numeric)

        spotify_profile = None
        if spotify_provider is None:
            spotify_provider = SpotifyPopularityProvider.get_shared()
        if spotify_provider and getattr(spotify_provider, "is_enabled", False):
            spotify_profile = spotify_provider.get_track_profile(track)

        spotify_popularity = None
        if isinstance(spotify_profile, dict):
            spotify_popularity = spotify_profile.get("popularity")

        fallback_reason = None
        base_score = None

        if normalized_rating is not None and spotify_popularity is not None:
            base_score = normalized_rating * spotify_popularity
        elif spotify_popularity is not None:
            base_score = spotify_popularity
            fallback_reason = "spotify_only"
        elif normalized_rating is not None:
            base_score = normalized_rating * self._default_spotify_baseline
            fallback_reason = "album_only"
        else:
            fallback_reason = "no_data"

        components = {
            "album_rating_count": album_rating_count,
            "normalized_album_rating": normalized_rating,
            "spotify_popularity": spotify_popularity,
            "base_score": base_score,
            "fallback": fallback_reason,
        }

        updated_details = dict(details)
        updated_details["components"] = components

        if playlist_logger:
            playlist_logger.debug(
                "AllMusic composite for '%s' → components=%s",
                getattr(track, "title", "<unknown>"),
                components,
            )

        return base_score, updated_details

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
    if rating_key in metadata_cache:
        return metadata_cache[rating_key]

    url = f"{PLEX_URL}/library/metadata/{rating_key}"
    headers = {"X-Plex-Token": PLEX_TOKEN}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    metadata_cache[rating_key] = response.text
    return metadata_cache[rating_key]

def parse_field_from_xml(xml_text, field):
    """Extract a field (attribute or tag) from Plex XML metadata."""
    try:
        root = ET.fromstring(xml_text)
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

# ----------------------------
# Attribute Retrieval (with Cache)
# ----------------------------
def get_field_value(track, field):
    """Retrieve and merge a field (e.g., genres) from track, album, and artist levels with caching."""
    values = set()  # use a set to avoid duplicates

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

    def _normalize_to_list(raw_value):
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
                normalized.extend(_normalize_to_list(item))
                continue

            candidate = getattr(item, "tag", item)
            if isinstance(candidate, (list, tuple, set)):
                normalized.extend(_normalize_to_list(candidate))
                continue

            text = str(candidate).strip()
            if text:
                normalized.append(text)
            elif candidate in {0, 0.0}:
                normalized.append(str(candidate))

        return normalized

    # Helper to extract and normalize field values
    def extract_values(source_key, field_name):
        if not source_key:
            return []
        try:
            xml_text = fetch_full_metadata(source_key)
            xml_val = parse_field_from_xml(xml_text, field_name)
            if xml_val is not None:
                return _normalize_to_list(xml_val)
        except Exception as e:
            logger.debug(f"Metadata error for key {source_key}: {e}")
        return []

    # Album type needs to collate information explicitly from album objects
    if field == "album.type":
        parent_key = getattr(track, "parentRatingKey", None)
        if parent_key:
            values.update(extract_values(parent_key, "type"))

        parent_type = getattr(track, "parentType", None)
        values.update(_normalize_to_list(parent_type))

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
            values.update(_normalize_to_list(getattr(album, "type", None)))

        return sorted(values)

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
            values.update(_normalize_to_list(val))

        if merge_styles_for_genres:
            track_styles = getattr(track, "styles", None)
            if track_styles is not None:
                if callable(track_styles):
                    try:
                        track_styles = track_styles()
                    except TypeError:
                        track_styles = None
                values.update(_normalize_to_list(track_styles))

        # 2️⃣ Try cached XML for the track
        track_key = getattr(track, "ratingKey", None)
        if track_key:
            values.update(extract_values(track_key, candidate))
            if merge_styles_for_genres and "track" not in style_sources_collected:
                values.update(extract_values(track_key, "styles"))
                style_sources_collected.add("track")

        # 3️⃣ Try album level
        parent_key = getattr(track, "parentRatingKey", None)
        if parent_key:
            values.update(extract_values(parent_key, candidate))
            if merge_styles_for_genres and "album" not in style_sources_collected:
                values.update(extract_values(parent_key, "styles"))
                style_sources_collected.add("album")

        # 4️⃣ Try artist level
        artist_key = getattr(track, "grandparentRatingKey", None)
        if artist_key:
            values.update(extract_values(artist_key, candidate))
            if merge_styles_for_genres and "artist" not in style_sources_collected:
                values.update(extract_values(artist_key, "styles"))
                style_sources_collected.add("artist")

        return len(values) > before

    collected = False
    for candidate in field_candidates:
        if collect_from_candidate(candidate):
            collected = True

    if not collected and resolved_field != field:
        collect_from_candidate(field)

    # Return sorted list for consistency
    return sorted(values)

# ----------------------------
# Field Comparison
# ----------------------------
def check_condition(value, operator, expected, match_all=True):
    """Compare a field value using the given operator."""
    if value is None:
        return False

    if isinstance(expected, str) and "," in expected:
        expected_values = [v.strip() for v in expected.split(",")]
    elif isinstance(expected, list):
        expected_values = expected
    else:
        expected_values = [expected]

    values = value if isinstance(value, (list, tuple, set)) else [value]

    results = []
    for exp in expected_values:
        exp_str = str(exp).lower()

        if operator == "equals":
            results.append(any(str(v).lower() == exp_str for v in values))
        elif operator == "does_not_equal":
            results.append(all(str(v).lower() != exp_str for v in values))
        elif operator == "contains":
            results.append(any(exp_str in str(v).lower() for v in values))
        elif operator == "does_not_contain":
            results.append(all(exp_str not in str(v).lower() for v in values))
        elif operator == "greater_than":
            try:
                results.append(any(float(v) > float(exp) for v in values))
            except (ValueError, TypeError):
                results.append(False)
        elif operator == "less_than":
            try:
                results.append(any(float(v) < float(exp) for v in values))
            except (ValueError, TypeError):
                results.append(False)
        else:
            logger.warning(f"Unknown operator: {operator}")
            results.append(False)

    return all(results) if match_all else any(results)

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


def _run_playlist_build(name, config, log, playlist_handler, playlist_log_path):
    if playlist_handler:
        log.debug(
            "Per-playlist debug logging for '%s' → %s",
            name,
            playlist_log_path,
        )

    log.info(f"Building playlist: {name}")
    build_start = time.perf_counter()
    filters = config.get("plex_filter", [])
    library = plex.library.section(config.get("library", LIBRARY_NAME))
    limit = config.get("limit")
    sort_by = config.get("sort_by")
    cover_path = config.get("cover")
    resolved_sort_by = sort_by
    sort_desc_in_config = "sort_desc" in config
    sort_desc = config.get("sort_desc", True)
    spotify_provider = SpotifyPopularityProvider.get_shared()
    if sort_by == "popularity":
        if spotify_provider and spotify_provider.is_enabled:
            resolved_sort_by = "spotifyPopularity"
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "Sort field '%s' will use Spotify popularity metrics",
                    sort_by,
                )
        else:
            error_detail = spotify_provider.describe_error() if spotify_provider else None
            log.warning(
                "Popularity sorting requested but Spotify is unavailable%s",
                f" ({error_detail})" if error_detail else "",
            )
            resolved_sort_by = None

    if sort_by == "alphabetical":
        resolved_sort_by = "__alphabetical__"
        if not sort_desc_in_config:
            sort_desc = False
    elif sort_by == "reverse_alphabetical":
        resolved_sort_by = "__alphabetical__"
        if not sort_desc_in_config:
            sort_desc = True
    elif sort_by == "oldest_first":
        resolved_sort_by = "originallyAvailableAt"
        if not sort_desc_in_config:
            sort_desc = False
    elif sort_by == "newest_first":
        resolved_sort_by = "originallyAvailableAt"
        if not sort_desc_in_config:
            sort_desc = True
    chunk_size = config.get("chunk_size", PLAYLIST_CHUNK_SIZE)
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

    fetch_start = time.perf_counter()
    all_tracks = library.searchTracks()
    total_tracks = len(all_tracks)
    progress_reporter = FilteringProgressReporter(log, name, total_tracks)
    fetch_duration = time.perf_counter() - fetch_start
    log.info(
        "Fetched %s tracks from %s in %.2fs",
        total_tracks,
        config.get("library", LIBRARY_NAME),
        fetch_duration,
    )

    try:
        existing = plex.playlist(name)
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
                playlist_obj = plex.createPlaylist(name, items=list(stream_buffer))
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
            keep = True
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
            for f in filters:
                field = f["field"]
                operator = f["operator"]
                expected = f["value"]
                match_all = f.get("match_all", True)

                val = get_field_value(track, field)
                if debug_logging:
                    log.debug(
                        "  Condition: field='%s', operator='%s', expected=%s, match_all=%s",
                        field,
                        operator,
                        expected,
                        match_all
                    )
                    log.debug("    Extracted value: %s", val)
                if not check_condition(val, operator, expected, match_all):
                    if debug_logging:
                        log.debug("    ❌ Condition failed for field '%s'", field)
                    keep = False
                    break
            if keep:
                match_count += 1
                if stream_enabled:
                    stream_buffer.append(track)
                    if len(stream_buffer) >= chunk_size:
                        flush_stream_buffer()
                else:
                    matched_tracks.append(track)
                if debug_logging:
                    log.debug("    ✅ Track matched all conditions")
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
            "Performance summary for '%s': fetch=%.2fs, filter=%.2fs (%.1f track/s), update=%.2fs, total=%.2fs",
            name,
            fetch_duration,
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
            spotify_provider=spotify_provider,
        )
        dedup_duration = time.perf_counter() - dedup_start
        if duplicates_removed:
            log.info(
                "Removed %s duplicate track(s) from playlist '%s' after popularity comparison",
                duplicates_removed,
                name,
            )

    album_popularity_cache = {}
    album_popularity_cache_by_object = {}
    if resolved_sort_by == "spotifyPopularity":
        (
            album_popularity_cache,
            album_popularity_cache_by_object,
        ) = _compute_album_popularity_boosts(
            matched_tracks,
            dedup_popularity_cache,
            spotify_provider=spotify_provider,
            playlist_logger=log,
        )
    match_count = len(matched_tracks)

    sort_duration = 0.0
    if resolved_sort_by:
        sort_value_cache = {}
        sort_value_cache_by_object = {}

        def _get_sort_value(track):
            cache_key = getattr(track, "ratingKey", None)
            cache_key_str = str(cache_key) if cache_key is not None else None
            object_cache_key = id(track)
            if cache_key_str and cache_key_str in sort_value_cache:
                return sort_value_cache[cache_key_str]

            if cache_key_str is None and object_cache_key in sort_value_cache_by_object:
                return sort_value_cache_by_object[object_cache_key]

            if cache_key_str and cache_key_str in dedup_popularity_cache:
                value = dedup_popularity_cache[cache_key_str]
                sort_value_cache[cache_key_str] = value
                return value

            if resolved_sort_by == "spotifyPopularity":
                popularity_value = None

                if cache_key_str and cache_key_str in album_popularity_cache:
                    popularity_value = album_popularity_cache[cache_key_str]
                elif cache_key_str is None and object_cache_key in album_popularity_cache_by_object:
                    popularity_value = album_popularity_cache_by_object[object_cache_key]
                else:
                    popularity_value = _resolve_track_popularity_value(
                        track,
                        spotify_provider=spotify_provider,
                        playlist_logger=log,
                    )

                if popularity_value is None:
                    sentinel = float("-inf") if sort_desc else float("inf")
                    if cache_key_str:
                        sort_value_cache[cache_key_str] = sentinel
                    else:
                        sort_value_cache_by_object[object_cache_key] = sentinel
                    return sentinel

                if cache_key_str:
                    sort_value_cache[cache_key_str] = popularity_value
                else:
                    sort_value_cache_by_object[object_cache_key] = popularity_value
                return popularity_value

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
            except Exception as exc:
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
            """Return a comparable representation for sorting, handling mixed types safely."""
            if value is None:
                return None
            if isinstance(value, tuple):
                return tuple(_normalize_sort_value(item) for item in value)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return value.lower()
            # Support datetime-like objects
            if hasattr(value, "timestamp"):
                try:
                    return float(value.timestamp())
                except Exception:
                    pass
            if hasattr(value, "isoformat"):
                try:
                    return value.isoformat()
                except Exception:
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
        matched_tracks.sort(key=cmp_to_key(_compare_tracks))
        sort_duration = time.perf_counter() - sort_start

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
            "Performance summary for '%s': fetch=%.2fs, filter=%.2fs (%.1f track/s), dedup=%.2fs, sort=%.2fs, update=%.2fs, total=%.2fs",
            name,
            fetch_duration,
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
                playlist_obj = plex.createPlaylist(name, items=chunk)
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
        "Performance summary for '%s': fetch=%.2fs, filter=%.2fs (%.1f track/s), dedup=%.2fs, sort=%.2fs, update=%.2fs (%.1f track/s), total=%.2fs",
        name,
        fetch_duration,
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
        SpotifyPopularityProvider.save_shared_cache()
        AllMusicPopularityProvider.save_shared_cache()
        if playlist_handler:
            log.debug(
                "Closing per-playlist debug logging for '%s'",
                name,
            )
            playlist_handler.close()

# ----------------------------
# Spotify Popularity Cache Builder
# ----------------------------
def build_spotify_popularity_cache():
    provider = SpotifyPopularityProvider.get_shared()

    if not SPOTIFY_CACHE_FILE:
        spotify_logger.warning(
            "Spotify popularity cache file is disabled via configuration; skipping population run.",
        )
        return {
            "status": "disabled",
            "reason": "cache_disabled",
        }

    if not provider.is_enabled:
        reason = provider.describe_error() or "Spotify integration is not configured."
        spotify_logger.warning("Spotify popularity cache build skipped: %s", reason)
        return {
            "status": "skipped",
            "reason": reason,
        }

    try:
        library = plex.library.section(LIBRARY_NAME)
    except Exception as exc:
        spotify_logger.exception("Unable to access Plex library for Spotify cache build: %s", exc)
        return {
            "status": "error",
            "reason": str(exc),
        }

    def _coerce_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    resume_state = provider.get_population_resume_state() or {}
    pending_rating_keys = []
    pending_entries = []
    resuming = False

    cached_profiles = 0
    new_profiles = 0
    refreshed_profiles = 0
    missing_profiles = 0
    errors = 0
    processed_pending = 0
    total_tracks = None

    if resume_state and resume_state.get("pending_keys"):
        resuming = True
        pending_rating_keys = [
            str(key)
            for key in resume_state.get("pending_keys", [])
            if key not in (None, "")
        ]
        cached_profiles = _coerce_int(resume_state.get("cached_profiles"), 0)
        new_profiles = _coerce_int(resume_state.get("new_profiles"), 0)
        refreshed_profiles = _coerce_int(resume_state.get("refreshed_profiles"), 0)
        missing_profiles = _coerce_int(resume_state.get("missing_profiles"), 0)
        errors = _coerce_int(resume_state.get("errors"), 0)
        processed_pending = _coerce_int(resume_state.get("processed"), 0)
        total_tracks = resume_state.get("total_tracks")
        try:
            total_tracks = int(total_tracks) if total_tracks is not None else None
        except (TypeError, ValueError):
            total_tracks = None

        if total_tracks is None:
            total_tracks = len(pending_rating_keys) + processed_pending + cached_profiles

        spotify_logger.info(
            "Resuming Spotify popularity cache build with %d pending track(s)",
            len(pending_rating_keys),
        )
    else:
        try:
            all_tracks = library.searchTracks()
        except Exception as exc:
            spotify_logger.exception("Unable to enumerate library tracks for Spotify cache build: %s", exc)
            return {
                "status": "error",
                "reason": str(exc),
            }

        total_tracks = len(all_tracks)
        spotify_logger.info("Starting Spotify popularity cache build for %d track(s)", total_tracks)

        for track in all_tracks:
            rating_key = getattr(track, "ratingKey", None)
            if rating_key is None:
                errors += 1
                spotify_logger.debug(
                    "Skipping track without rating key during Spotify cache build: %s",
                    getattr(track, "title", "<unknown>"),
                )
                continue

            _, had_cache, was_stale = provider.inspect_track_cache(track)
            if had_cache and not was_stale:
                cached_profiles += 1
                continue

            pending_entries.append((str(rating_key), track))

        pending_rating_keys = [key for key, _ in pending_entries]
        provider.begin_population_run(pending_rating_keys, total_tracks, cached_profiles)

        if pending_rating_keys:
            spotify_logger.info(
                "Spotify popularity cache build will fetch %d uncached track(s); %d cached value(s) reused.",
                len(pending_rating_keys),
                cached_profiles,
            )
        else:
            spotify_logger.info(
                "Spotify popularity cache build found %d cached track(s); nothing to fetch.",
                cached_profiles,
            )

    total_pending_entries = len(pending_rating_keys)

    chunk_size = SPOTIFY_POPULATION_CHUNK_SIZE or 0
    chunk_delay = SPOTIFY_POPULATION_CHUNK_DELAY or 0.0
    if chunk_size and total_pending_entries:
        if chunk_delay > 0:
            spotify_logger.info(
                "Spotify popularity cache build will pause for %.2fs after every %d track(s) to reduce Spotify load.",
                chunk_delay,
                chunk_size,
            )
        else:
            spotify_logger.info(
                "Spotify popularity cache build will process tracks in chunks of %d before continuing immediately.",
                chunk_size,
            )

    initial_cached_progress = cached_profiles
    initial_progress = initial_cached_progress + processed_pending

    if total_tracks is None:
        total_tracks = len(pending_rating_keys) + initial_cached_progress
    if total_tracks < initial_progress:
        total_tracks = initial_progress

    progress_bar = tqdm(
        total=total_tracks,
        desc="Caching Spotify popularity",
        unit="track",
        dynamic_ncols=True,
    )

    if initial_progress:
        progress_bar.update(initial_progress)

    if SAVE_INTERVAL and initial_progress and initial_progress % SAVE_INTERVAL == 0:
        provider.save_cache()

    start_time = time.perf_counter()

    initial_processed_counter = processed_pending

    if resuming:
        def _iter_pending():
            for rating_key in pending_rating_keys:
                track_obj = None
                try:
                    fetch_key = int(rating_key)
                except (TypeError, ValueError):
                    fetch_key = rating_key
                try:
                    track_obj = plex.fetchItem(fetch_key)
                except Exception as exc:
                    spotify_logger.debug(
                        "Unable to fetch track %s from Plex while resuming Spotify cache build: %s",
                        rating_key,
                        exc,
                    )
                yield rating_key, track_obj

        pending_iterator = _iter_pending()
    else:
        pending_iterator = iter(pending_entries)

    try:
        for rating_key, track in pending_iterator:
            rating_key_str = None if rating_key in (None, "") else str(rating_key)
            outcome = None

            try:
                if track is None:
                    errors += 1
                    outcome = "error"
                else:
                    _, had_cache, was_stale = provider.inspect_track_cache(track)
                    if had_cache and not was_stale:
                        cached_profiles += 1
                        outcome = "cached"
                    else:
                        profile = provider.get_track_profile(track)
                        if profile is None:
                            missing_profiles += 1
                            outcome = "missing"
                        else:
                            if had_cache:
                                refreshed_profiles += 1
                                outcome = "refreshed"
                            else:
                                new_profiles += 1
                                outcome = "new"
            except Exception as exc:
                errors += 1
                track_title = getattr(track, "title", "<unknown>") if track else "<missing>"
                track_artist = getattr(track, "grandparentTitle", "<unknown>") if track else "<missing>"
                spotify_logger.debug(
                    "Failed to populate Spotify popularity for '%s' by '%s': %s",
                    track_title,
                    track_artist,
                    exc,
                )
                outcome = "error"
            finally:
                processed_pending += 1
                if rating_key_str is not None:
                    provider.update_population_resume(rating_key_str, outcome or "error")

                total_progress = initial_cached_progress + processed_pending
                if SAVE_INTERVAL and total_progress % SAVE_INTERVAL == 0:
                    provider.save_cache()

                progress_bar.update(1)

                if chunk_size:
                    processed_in_run = processed_pending - initial_processed_counter
                    remaining_in_run = max(total_pending_entries - processed_in_run, 0)
                    if processed_pending % chunk_size == 0 and remaining_in_run > 0:
                        provider.save_cache()
                        chunk_processed_run = processed_in_run
                        chunk_processed_total = processed_pending
                        if chunk_delay > 0:
                            spotify_logger.info(
                                "Chunk boundary reached (%d processed this run, %d total); %d track(s) remain. "
                                "Sleeping for %.2fs to respect Spotify rate limits.",
                                chunk_processed_run,
                                chunk_processed_total,
                                remaining_in_run,
                                chunk_delay,
                            )
                            time.sleep(chunk_delay)
                        else:
                            spotify_logger.info(
                                "Chunk boundary reached (%d processed this run, %d total); %d track(s) remain.",
                                chunk_processed_run,
                                chunk_processed_total,
                                remaining_in_run,
                            )
    finally:
        progress_bar.close()

    provider.save_cache()

    duration = time.perf_counter() - start_time
    processed_tracks = initial_cached_progress + processed_pending

    summary = {
        "status": "completed",
        "total_tracks": total_tracks,
        "processed_tracks": processed_tracks,
        "new_profiles": new_profiles,
        "refreshed_profiles": refreshed_profiles,
        "cached_profiles": cached_profiles,
        "missing_profiles": missing_profiles,
        "errors": errors,
        "duration_seconds": round(duration, 2),
    }

    timestamp = provider.record_population_run(summary)
    provider.save_cache()

    spotify_logger.info(
        "Spotify popularity cache build complete: new=%d refreshed=%d cached=%d missing=%d errors=%d (%.2fs)",
        new_profiles,
        refreshed_profiles,
        cached_profiles,
        missing_profiles,
        errors,
        duration,
    )
    spotify_logger.info("Spotify popularity cache metadata updated at %s", timestamp)

    return summary


# ----------------------------
# Cache-only Mode with Resume
# ----------------------------
def build_metadata_cache():
    logger.info("Starting cache-only metadata build (resumable)...")
    library = plex.library.section(LIBRARY_NAME)
    all_tracks = library.searchTracks()
    total_tracks = len(all_tracks)
    logger.info(f"Found {total_tracks} tracks in library '{LIBRARY_NAME}'")

    processed = 0
    with tqdm(total=total_tracks, desc="Caching metadata", unit="track", dynamic_ncols=True) as pbar:
        for track in all_tracks:
            # Skip already-cached items
            keys_to_fetch = [track.ratingKey]
            if hasattr(track, "parentRatingKey"):
                keys_to_fetch.append(track.parentRatingKey)
            if hasattr(track, "grandparentRatingKey"):
                keys_to_fetch.append(track.grandparentRatingKey)

            new_keys = [k for k in keys_to_fetch if k not in metadata_cache]
            if not new_keys:
                pbar.update(1)
                continue

            for key in new_keys:
                try:
                    fetch_full_metadata(key)
                except Exception as e:
                    logger.warning(f"Failed to fetch {key}: {e}")
            processed += 1
            if processed % SAVE_INTERVAL == 0:
                save_cache()
            pbar.update(1)

    save_cache()
    logger.info("✅ Metadata caching complete and saved to disk.")

# ----------------------------
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
        "--build-popularity-cache",
        action="store_true",
        help="Populate the Spotify popularity cache and exit.",
    )
    parser.add_argument(
        "--playlist",
        dest="playlists",
        action="append",
        help="Name of a playlist to build. Can be provided multiple times.",
    )
    args = parser.parse_args()

    if args.build_popularity_cache:
        try:
            summary = build_spotify_popularity_cache()
        except Exception as exc:
            spotify_logger.exception("Spotify popularity cache build failed: %s", exc)
            sys.exit(1)

        status = (summary or {}).get("status")
        reason = (summary or {}).get("reason")

        if status == "completed":
            spotify_logger.info("Spotify popularity cache build finished successfully.")
            sys.exit(0)
        elif status in {"disabled", "skipped"}:
            if reason:
                spotify_logger.info("Spotify popularity cache build %s: %s", status, reason)
            else:
                spotify_logger.info("Spotify popularity cache build %s.", status)
            sys.exit(0)
        else:
            if reason:
                spotify_logger.error(
                    "Spotify popularity cache build %s: %s", status or "failed", reason
                )
            else:
                spotify_logger.error("Spotify popularity cache build failed.")
            sys.exit(1)

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
