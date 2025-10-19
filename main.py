import os
import re
import yaml
import math
import logging
import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cmp_to_key
from xml.etree import ElementTree as ET
from html import unescape
import unicodedata
from urllib.parse import quote_plus

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

cfg = load_yaml("/app/config.yml")

PLAYLISTS_FILE = "/app/playlists.yml"
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

spotify_cfg = cfg.get("spotify", {}) or {}
SPOTIFY_CLIENT_ID = spotify_cfg.get("client_id")
SPOTIFY_CLIENT_SECRET = spotify_cfg.get("client_secret")
SPOTIFY_MARKET = spotify_cfg.get("market")
SPOTIFY_SEARCH_LIMIT = spotify_cfg.get("search_limit", 10)

allmusic_cfg = cfg.get("allmusic", {}) or {}
ALLMUSIC_ENABLED = allmusic_cfg.get("enabled", True)
ALLMUSIC_CACHE_FILE = allmusic_cfg.get("cache_file", "/app/allmusic_popularity.json")
ALLMUSIC_TIMEOUT = allmusic_cfg.get("timeout", 10)
ALLMUSIC_USER_AGENT = allmusic_cfg.get(
    "user_agent",
    "plex-music-builder/1.0 (+https://github.com/plexmusicbuilder)",
)
ALLMUSIC_CACHE_VERSION = 3

logging_cfg = cfg.get("logging", {})
LOG_LEVEL = logging_cfg.get("level", "DEBUG").upper()
LOG_FILE = logging_cfg.get("file", "/app/logs/plex_music_builder.log")
ACTIVE_LOG_FILE = None

_default_log_dir = os.path.dirname(LOG_FILE) if LOG_FILE else ""
if not _default_log_dir:
    _default_log_dir = "/app/logs"

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
    logger_obj.addHandler(stream_handler)

    if LOG_FILE:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        try:
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        except OSError as exc:
            logger_obj.error(f"Unable to open log file '{LOG_FILE}': {exc}")
        else:
            file_handler.setFormatter(formatter)
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
playlists_data = raw_playlists.get("playlists", {}) if isinstance(raw_playlists, dict) else {}

# ----------------------------
# Metadata Cache
# ----------------------------
CACHE_FILE = "/app/metadata_cache.json"
METADATA_PROVIDER_URL = "https://metadata.provider.plex.tv"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        try:
            metadata_cache = json.load(f)
        except json.JSONDecodeError:
            metadata_cache = {}
else:
    metadata_cache = {}

# Cache canonical/original release metadata fetched from Plex's metadata provider.
# The cache is keyed by the normalized album GUID (with the ``plex://`` prefix
# removed) and stores the raw XML payload plus any fields we've already parsed
# from that payload.
canonical_metadata_cache = {}
canonical_guid_lookup_cache = {}
canonical_cache_lock = threading.Lock()

# Canonical metadata exposes album-level attributes (e.g., original release
# year, popularity counts). Map the fields we request in Plex filters/sorts to
# their canonical equivalents so we don't have to special case them scattered
# throughout the codebase.
CANONICAL_FIELD_MAP = {
    "parentGuid": "guid",
    "guid": "guid",
    "parentYear": "year",
    "year": "year",
    "originallyAvailableAt": "originallyAvailableAt",
    "ratingCount": "ratingCount",
    "parentRatingCount": "ratingCount",
}

CANONICAL_SORT_FIELDS = {"ratingCount", "parentRatingCount", "year", "parentYear", "originallyAvailableAt"}
CANONICAL_NUMERIC_FIELDS = {"ratingCount", "parentRatingCount", "year", "parentYear"}
CANONICAL_ONLY_FIELDS = {"ratingCount", "parentRatingCount"}

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_cache, f)


def _normalize_plex_guid(guid):
    if not guid:
        return None
    guid_str = str(guid).strip()
    if not guid_str:
        return None
    if guid_str.startswith("plex://"):
        return guid_str[len("plex://") :]
    return guid_str or None


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


def _build_canonical_identity_key(artist, album, year):
    artist_key = _normalize_compare_value(artist)
    album_key = _normalize_compare_value(album)
    year_key = str(year).strip() if year else ""
    return (artist_key, album_key, year_key)


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


def _search_canonical_guid(artist, album, year):
    log = _get_active_logger()
    if not album:
        return None

    if CACHE_ONLY:
        return None

    stripped_album = _strip_parenthetical(album)
    search_album = (stripped_album or "").strip()

    if not search_album:
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "Canonical search skipping album '%s' by '%s' because sanitized name is empty",
                album,
                artist,
            )
        return None

    album_query_candidates = []
    if search_album:
        album_query_candidates.append(search_album)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "Canonical search normalized album '%s' to '%s' for artist='%s' year='%s'",
            album,
            search_album,
            artist,
            year,
        )

    query_strings = []
    for base in album_query_candidates or [search_album]:
        query_strings.append(base)
        if artist:
            query_strings.append(f"{base} {artist}")
        if year:
            query_strings.append(f"{base} {year}")
            if artist:
                query_strings.append(f"{base} {artist} {year}")

    deduped_queries = []
    seen = set()
    for query in query_strings:
        normalized = query.strip().lower()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped_queries.append(query.strip())

    if not deduped_queries and search_album:
        deduped_queries = [search_album]

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "Canonical search queries for '%s' by '%s' (%s): %s",
            search_album,
            artist,
            year,
            deduped_queries,
        )

    album_norm_inputs = []
    if album:
        album_norm_inputs.append(album)
    if stripped_album and stripped_album != album:
        album_norm_inputs.append(stripped_album)
    if not album_norm_inputs:
        album_norm_inputs = [search_album]

    album_norms = _collect_normalized_candidates(album_norm_inputs)
    artist_norms = _collect_normalized_candidates([artist])
    year_token = _extract_year_token(year)

    def _build_endpoint(base, path):
        base = (base or "").rstrip("/")
        return f"{base}{path}" if base else None

    endpoint_variants = []

    def _append_variant(base, path, extra_params):
        endpoint = _build_endpoint(base, path)
        if not endpoint:
            return
        variant_key = (endpoint, tuple(sorted(extra_params.items())))
        if variant_key in seen_variants:
            return
        seen_variants.add(variant_key)
        endpoint_variants.append((endpoint, extra_params))

    seen_variants = set()

    search_paths = [
        "/library/metadata/search",
        "/library/search",
        "/search",
        "/hubs/search",
    ]

    param_variants = [
        {"type": 9},
        {"searchTypes": "albums"},
        {"searchTypes": "album"},
        {"includeCollections": 1, "includeExternalMedia": 1, "searchTypes": "album"},
        {"includeCollections": 1, "includeExternalMedia": 1},
        {},
    ]

    # Prefer searching against the user's own Plex server first so that we stop
    # iterating as soon as we find a result.  The metadata provider endpoints
    # frequently respond with ``404`` for user specific tokens which produced a
    # lot of noisy debug logging even though the subsequent local search would
    # succeed.  By prioritizing ``PLEX_URL`` we usually resolve the GUID before
    # hitting the remote service while still keeping it available as a
    # fallback.
    for base in (PLEX_URL, METADATA_PROVIDER_URL):
        for path in search_paths:
            for extra_params in param_variants:
                # ``type`` is not valid for ``/hubs/search`` endpoints; skip to avoid noisy logs.
                if path == "/hubs/search" and "type" in extra_params:
                    continue
                # ``searchTypes`` is not valid for legacy metadata endpoints; they expect ``type``
                # or no extra parameters. Keep the combinations flexible but avoid duplicates.
                if path in {"/library/metadata/search", "/library/search"}:
                    if extra_params.get("searchTypes") == "album":
                        continue
                    if extra_params.get("includeCollections"):
                        continue
                _append_variant(base, path, extra_params)

    search_endpoints = endpoint_variants

    for query in deduped_queries:
        for endpoint, extra_params in search_endpoints:
            params = {
                "query": query,
                "X-Plex-Token": PLEX_TOKEN,
            }
            params.update(extra_params)
            params = {k: v for k, v in params.items() if v is not None}
            debug_params = {k: v for k, v in params.items() if k != "X-Plex-Token"}

            try:
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        "Canonical search requesting '%s' from %s with params %s",
                        query,
                        endpoint,
                        debug_params,
                    )
                response = requests.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
            except requests.RequestException as exc:
                log.debug(
                    "Canonical metadata search failed for query '%s' via %s with params %s: %s",
                    query,
                    endpoint,
                    debug_params,
                    exc,
                )
                continue

            guid = _extract_guid_from_search_results(
                response.text,
                album_norms,
                artist_norms,
                year_token,
            )
            if guid:
                log.debug(
                    "Canonical metadata search resolved '%s' by '%s' (%s) to GUID %s",
                    album,
                    artist,
                    year,
                    _normalize_plex_guid(guid),
                )
                return guid

    log.debug(
        "Canonical metadata search could not resolve '%s' by '%s' (%s)",
        album,
        artist,
        year,
    )
    return None


def _lookup_canonical_guid_for_track(track):
    artist = getattr(track, "grandparentTitle", None)
    album = getattr(track, "parentTitle", None)
    year = _resolve_album_year(track)

    identity_key = _build_canonical_identity_key(artist, album, year)

    with canonical_cache_lock:
        if identity_key in canonical_guid_lookup_cache:
            cached_guid = canonical_guid_lookup_cache[identity_key]
            return cached_guid or None

    resolved_guid = _search_canonical_guid(artist, album, year)

    with canonical_cache_lock:
        canonical_guid_lookup_cache[identity_key] = resolved_guid or ""

    return resolved_guid


def fetch_canonical_album_metadata(album_guid):
    """Fetch the canonical/original release metadata for an album GUID."""

    normalized_guid = _normalize_plex_guid(album_guid)
    if not normalized_guid:
        return None

    with canonical_cache_lock:
        cached = canonical_metadata_cache.get(normalized_guid)
        if cached:
            return cached.get("xml")

    url = f"{METADATA_PROVIDER_URL}/library/metadata/{normalized_guid}"
    params = {"X-Plex-Token": PLEX_TOKEN}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        xml_text = response.text
    except requests.RequestException as exc:
        logger.debug(
            "Failed to fetch canonical metadata for album GUID '%s': %s",
            normalized_guid,
            exc,
        )
        xml_text = None

    with canonical_cache_lock:
        canonical_metadata_cache[normalized_guid] = {"xml": xml_text, "fields": {}}

    return xml_text


def get_canonical_field_from_guid(album_guid, field):
    """Return a canonical metadata field for an album GUID, if available."""

    normalized_guid = _normalize_plex_guid(album_guid)
    if not normalized_guid:
        return None

    with canonical_cache_lock:
        cached = canonical_metadata_cache.get(normalized_guid)
        if cached and field in cached.get("fields", {}):
            return cached["fields"][field]
        xml_text = cached.get("xml") if cached else None

    if xml_text is None:
        xml_text = fetch_canonical_album_metadata(album_guid)
        if xml_text is None:
            return None

    value = parse_field_from_xml(xml_text, field)

    with canonical_cache_lock:
        cached = canonical_metadata_cache.setdefault(
            normalized_guid, {"xml": xml_text, "fields": {}}
        )
        cached["fields"][field] = value

    return value


def get_canonical_field_for_track(track, field):
    """Resolve a canonical/original-release field for the provided track."""

    canonical_field = CANONICAL_FIELD_MAP.get(field)
    if not canonical_field:
        return None

    guid = None

    if field == "ratingCount":
        guid = _get_track_guid(track)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Attempting canonical %s lookup for track '%s' via track GUID %s",
                field,
                getattr(track, "title", "<unknown>"),
                _normalize_plex_guid(guid),
            )
        if guid:
            value = get_canonical_field_from_guid(guid, canonical_field)
            if value is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Resolved canonical %s=%s from track GUID %s",
                        field,
                        value,
                        _normalize_plex_guid(guid),
                    )
                return value

    if guid is None:
        guid = _get_album_guid(track)
        if not guid:
            return None

    if logger.isEnabledFor(logging.DEBUG) and field == "ratingCount":
        logger.debug(
            "Falling back to album GUID %s for canonical %s lookup on track '%s'",
            _normalize_plex_guid(guid),
            field,
            getattr(track, "title", "<unknown>"),
        )

    value = get_canonical_field_from_guid(guid, canonical_field)
    if value is not None:
        return value

    lookup_guid = _lookup_canonical_guid_for_track(track)
    if lookup_guid:
        lookup_normalized = _normalize_plex_guid(lookup_guid)
        guid_normalized = _normalize_plex_guid(guid)
        if lookup_normalized != guid_normalized:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Retrying canonical %s lookup for '%s' via searched GUID %s",
                    field,
                    getattr(track, "title", "<unknown>"),
                    lookup_normalized,
                )
            return get_canonical_field_from_guid(lookup_guid, canonical_field)

    return value


class SpotifyPopularityProvider:
    """Shared Spotify client that resolves track popularity for sorting."""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._client = None
        self._client_lock = threading.Lock()
        self._track_cache = {}
        self._id_cache = {}
        self._query_cache = {}
        self._enabled = False
        self._error = None
        self.market = self._normalize_market(SPOTIFY_MARKET)
        self.search_limit = self._coerce_search_limit(SPOTIFY_SEARCH_LIMIT)

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
            logger.warning("Failed to initialize Spotify client: %s", exc)
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

    @classmethod
    def get_shared(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def is_enabled(self):
        return self._enabled and self._client is not None

    def describe_error(self):
        return self._error

    def get_popularity(self, track):
        profile = self.get_track_profile(track)
        if isinstance(profile, dict):
            return profile.get("popularity")
        return profile

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

        profile = None
        for spotify_id in self._iter_spotify_track_ids(track):
            profile = self._get_track_profile_by_id(spotify_id)
            if profile is not None:
                break

        if profile is None:
            profile = self._search_for_track_profile(track)

        if track_key is not None:
            self._track_cache[track_key] = profile
        return profile

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

    def _get_track_profile_by_id(self, spotify_id):
        if spotify_id in self._id_cache:
            cached = self._id_cache[spotify_id]
            if isinstance(cached, dict) or cached is None:
                return cached
            return {"popularity": cached}

        try:
            with self._client_lock:
                data = self._client.track(spotify_id)
        except SpotifyException as exc:
            logger.debug(
                "Spotify track lookup failed for %s: %s",
                spotify_id,
                exc,
            )
            self._id_cache[spotify_id] = None
            return None
        except Exception as exc:
            logger.debug(
                "Unexpected Spotify error during track lookup for %s: %s",
                spotify_id,
                exc,
            )
            self._id_cache[spotify_id] = None
            return None

        profile = self._extract_track_profile(data)
        self._id_cache[spotify_id] = profile
        return profile

    def _search_for_track_profile(self, track):
        query = self._build_search_query(track)
        if not query:
            return None

        cache_key = (query, self.market)
        if cache_key in self._query_cache:
            candidates = self._query_cache[cache_key]
        else:
            try:
                with self._client_lock:
                    response = self._client.search(
                        q=query,
                        type="track",
                        market=self.market,
                        limit=self.search_limit,
                    )
            except SpotifyException as exc:
                logger.debug("Spotify search failed for '%s': %s", query, exc)
                self._query_cache[cache_key] = []
                return None
            except Exception as exc:
                logger.debug("Unexpected Spotify error during search for '%s': %s", query, exc)
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
        self._cache_lock = threading.Lock()
        self._cache_dirty = False
        self._default_spotify_baseline = 50.0

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

        popularity_value, computed_details = self._compute_composite_popularity(
            track,
            details,
            spotify_provider=spotify_provider,
            playlist_logger=playlist_logger,
        )

        self._remember_track_cache(track_key, popularity_value, computed_details)
        return popularity_value

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
            search_url = f"https://www.allmusic.com/search/albums/{quote_plus(variant)}"

            if playlist_logger:
                playlist_logger.debug("AllMusic album search URL: %s", search_url)

            try:
                response = self._session.get(search_url, timeout=self._timeout)
                response.raise_for_status()
            except Exception as exc:
                last_error = exc
                self._error = str(exc)
                if playlist_logger:
                    playlist_logger.debug(
                        "AllMusic album search for '%s' failed: %s",
                        variant,
                        exc,
                    )
                else:
                    logger.debug("AllMusic album search for '%s' failed: %s", variant, exc)
                self._remember_query_cache_entry(variant, None)
                continue

            album_candidate = self._select_album_candidate(response.text, album_info)

            if not album_candidate:
                self._remember_query_cache_entry(variant, None)
                continue

            album_url = album_candidate.get("album_url")

            if playlist_logger and album_url:
                playlist_logger.debug("AllMusic album page URL: %s", album_url)

            metadata = self._get_album_metadata(
                album_url,
                playlist_logger=playlist_logger,
            )

            highlighted_tracks = metadata.get("highlighted_tracks") or []
            rating_count = metadata.get("rating_count")

            artists = list(album_candidate.get("artists", []))
            if not artists and artist:
                artists = [artist]

            cleaned_album_title = album_candidate.get("album") or (
                _strip_parenthetical(album) if album else None
            )

            details = {
                "title": title,
                "artists": artists,
                "album": cleaned_album_title or None,
                "album_url": album_url,
                "rating_count": rating_count,
                "album_rating_count": rating_count,
                "album_highlighted_tracks": list(highlighted_tracks),
                "source": "album_search",
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
            search_url = f"https://www.allmusic.com/search/albums/{quote_plus(variant)}"

            if playlist_logger:
                playlist_logger.debug("AllMusic album search URL: %s", search_url)
            else:
                logger.debug("AllMusic album search URL: %s", search_url)

            try:
                response = self._session.get(search_url, timeout=self._timeout)
                response.raise_for_status()
            except Exception as exc:
                last_error = exc
                if playlist_logger:
                    playlist_logger.debug(
                        "AllMusic album search for '%s' failed: %s",
                        variant,
                        exc,
                    )
                else:
                    logger.debug("AllMusic album search for '%s' failed: %s", variant, exc)
                continue

            album_candidate = self._select_album_candidate(
                response.text,
                {
                    "album": album,
                    "artist": artist,
                },
            )

            if not album_candidate:
                continue

            metadata = self._get_album_metadata(
                album_candidate.get("album_url"),
                playlist_logger=playlist_logger,
            )

            highlighted_tracks = metadata.get("highlighted_tracks") or []

            artists = list(album_candidate.get("artists", []))
            if not artists and artist:
                artists = [artist]

            details = {
                "title": track_title,
                "artists": artists,
                "album": album_candidate.get("album"),
                "album_url": album_candidate.get("album_url"),
                "rating_count": metadata.get("rating_count"),
                "album_rating_count": metadata.get("rating_count"),
                "album_highlighted_tracks": list(highlighted_tracks),
                "source": "album_search",
            }

            album_url = album_candidate.get("album_url")

            if playlist_logger:
                if album_url:
                    playlist_logger.debug("AllMusic album page URL: %s", album_url)
                playlist_logger.debug(
                    "AllMusic album fallback matched '%s' → album_url='%s'",
                    album_candidate.get("album"),
                    album_url,
                )
            else:
                if album_url:
                    logger.debug("AllMusic album page URL: %s", album_url)
                logger.debug(
                    "AllMusic album fallback matched '%s' → album_url='%s'",
                    album_candidate.get("album"),
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

        highlighted_tracks = album_metadata.get("highlighted_tracks") or []

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
            "album_highlighted_tracks": list(highlighted_tracks),
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
        highlighted_tracks = cls._parse_album_highlighted_tracks(html)

        metadata = {}
        if rating_count is not None:
            metadata["rating_count"] = rating_count
        if highlighted_tracks:
            metadata["highlighted_tracks"] = sorted(highlighted_tracks)

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

    @classmethod
    def _parse_album_highlighted_tracks(cls, html):
        if not html:
            return set()

        highlighted = set()

        row_pattern = re.compile(
            r"<tr[^>]*class=\"[^\"]*(?:track|song)[^\"]*\"[^>]*>(.*?)</tr>",
            re.IGNORECASE | re.DOTALL,
        )

        for block in row_pattern.findall(html):
            if "highlight" not in block.lower():
                continue

            title = cls._extract_text(block, r'class=\"title\"[^>]*>(.*?)</a>')
            if not title:
                title = cls._extract_text(block, r'class=\"title\"[^>]*>(.*?)</td>')

            if title:
                highlighted.add(title)

        cell_pattern = re.compile(
            r'<td[^>]*class=\"[^\"]*title[^\"]*highlight[^\"]*\"[^>]*>(.*?)</td>',
            re.IGNORECASE | re.DOTALL,
        )

        for block in cell_pattern.findall(html):
            text = re.sub(r"<[^>]+>", " ", block)
            text = unescape(text).strip()
            if text:
                highlighted.add(text)

        return highlighted

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

        is_single = self._detect_single(
            track,
            spotify_profile,
            allmusic_details=details,
        )
        multiplier = 2.0 if is_single else 1.0

        final_score = None
        if base_score is not None:
            final_score = base_score * multiplier

        components = {
            "album_rating_count": album_rating_count,
            "normalized_album_rating": normalized_rating,
            "spotify_popularity": spotify_popularity,
            "base_score": base_score,
            "is_single": is_single,
            "single_multiplier": multiplier,
            "fallback": fallback_reason,
            "final_score": final_score,
        }

        updated_details = dict(details)
        updated_details["components"] = components

        if playlist_logger:
            playlist_logger.debug(
                "AllMusic composite for '%s' → components=%s",
                getattr(track, "title", "<unknown>"),
                components,
            )

        return final_score, updated_details

    def _detect_single(self, track, spotify_profile, allmusic_details=None):
        if isinstance(allmusic_details, dict):
            highlighted_tracks = allmusic_details.get("album_highlighted_tracks")
            if highlighted_tracks:
                track_title_norm = _normalize_compare_value(getattr(track, "title", None))
                highlighted_norms = _collect_normalized_candidates(highlighted_tracks)
                if track_title_norm and track_title_norm in highlighted_norms:
                    return True

        if isinstance(spotify_profile, dict):
            album_info = spotify_profile.get("album") or {}
            album_type = (album_info.get("album_type") or "").lower()
            album_group = (album_info.get("album_group") or "").lower()
            if album_type == "single" or album_group == "single":
                return True
            if isinstance(spotify_profile.get("is_single"), bool) and spotify_profile.get("is_single"):
                return True
            track_number = spotify_profile.get("track_number")
            if album_type == "ep" and track_number == 1:
                return True

        release_type = getattr(track, "subtype", None) or getattr(track, "type", None)
        if isinstance(release_type, str) and release_type.lower() == "single":
            return True

        return False

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
        # library lookups, ``Track`` for individual items, ``Metadata`` for the
        # canonical/original-release provider). Search a list of known element
        # names so we can reuse this helper for all of them.
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

        # Special handling for genres
        if field == "genres":
            # Return a list of genre strings
            genres = [el.attrib.get("tag").strip() for el in root.findall(".//Genre") if "tag" in el.attrib]
            if genres:
                return genres  # <- return list, no join

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

    # Helper to extract and normalize field values
    def extract_values(source_key, field_name):
        if not source_key:
            return []
        try:
            xml_text = fetch_full_metadata(source_key)
            xml_val = parse_field_from_xml(xml_text, field_name)
            if xml_val:
                # normalize to string list
                if isinstance(xml_val, (list, set)):
                    return [str(v).strip() for v in xml_val]
                return [str(xml_val).strip()]
        except Exception as e:
            logger.debug(f"Metadata error for key {source_key}: {e}")
        return []

    def extract_canonical_values(field_name):
        if field_name not in CANONICAL_FIELD_MAP:
            return []
        canonical_value = get_canonical_field_for_track(track, field_name)
        if canonical_value is None:
            return []
        if isinstance(canonical_value, (list, set)):
            return [str(v).strip() for v in canonical_value if str(v).strip()]
        value_str = str(canonical_value).strip()
        return [value_str] if value_str else []

    seen_fields = set()

    def collect_from_candidate(candidate):
        if not candidate or candidate in seen_fields:
            return False
        seen_fields.add(candidate)

        if candidate in CANONICAL_ONLY_FIELDS:
            canonical_value = get_canonical_field_for_track(track, candidate)
            if canonical_value is None:
                return False

            if isinstance(canonical_value, (list, set, tuple)):
                normalized_values = [str(v).strip() for v in canonical_value if str(v).strip()]
            else:
                normalized = str(canonical_value).strip()
                normalized_values = [normalized] if normalized or canonical_value == 0 else []

            values.update(normalized_values)
            return bool(normalized_values)

        before = len(values)

        # 1️⃣ Try direct object field (most accurate)
        val = getattr(track, candidate, None)
        if val or val == 0:
            if callable(val):
                try:
                    val = val()
                except TypeError:
                    val = None
            if val is not None:
                if isinstance(val, list):
                    values.update(str(v).strip() for v in val)
                else:
                    values.add(str(val).strip())

        # 2️⃣ Try cached XML for the track
        values.update(extract_values(track.ratingKey, candidate))

        # 3️⃣ Try album level
        parent_key = getattr(track, "parentRatingKey", None)
        if parent_key:
            values.update(extract_values(parent_key, candidate))

        # 4️⃣ Try canonical/original album metadata
        values.update(extract_canonical_values(candidate))

        # 5️⃣ Try artist level
        artist_key = getattr(track, "grandparentRatingKey", None)
        if artist_key:
            values.update(extract_values(artist_key, candidate))

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


def _run_playlist_build(name, config, log, playlist_handler, playlist_log_path):
    if playlist_handler:
        log.debug(
            "Per-playlist debug logging for '%s' → %s",
            name,
            playlist_log_path,
        )

    log.info(f"Building playlist: {name}")
    filters = config.get("plex_filter", [])
    library = plex.library.section(config.get("library", LIBRARY_NAME))
    limit = config.get("limit")
    sort_by = config.get("sort_by")
    cover_path = config.get("cover")
    resolved_sort_by = sort_by
    spotify_provider = None
    allmusic_provider = None
    if sort_by == "popularity":
        spotify_provider = SpotifyPopularityProvider.get_shared()
        allmusic_provider = AllMusicPopularityProvider.get_shared()

        def _log_popularity_source(message, *args):
            if log.isEnabledFor(logging.DEBUG):
                log.debug(message, *args)

        if allmusic_provider and allmusic_provider.is_enabled:
            resolved_sort_by = "spotifyPopularity"
            _log_popularity_source(
                "Sort field '%s' will use AllMusic × Spotify composite popularity",
                sort_by,
            )
            if spotify_provider and spotify_provider.is_enabled:
                _log_popularity_source(
                    "Spotify popularity fallback is enabled for playlist '%s'",
                    name,
                )
            elif spotify_provider:
                error_detail = spotify_provider.describe_error()
                if error_detail:
                    _log_popularity_source(
                        "Spotify popularity fallback unavailable (%s)",
                        error_detail,
                    )
        elif spotify_provider and spotify_provider.is_enabled:
            resolved_sort_by = "spotifyPopularity"
            _log_popularity_source(
                "AllMusic popularity disabled; using Spotify track popularity",
                sort_by,
            )
        else:
            error_detail = spotify_provider.describe_error() if spotify_provider else None
            log.warning(
                "Popularity sorting requested but no providers are available%s",
                f" ({error_detail})" if error_detail else "",
            )
            resolved_sort_by = None
    chunk_size = config.get("chunk_size", PLAYLIST_CHUNK_SIZE)
    stream_requested = config.get("stream_while_filtering", False)
    stream_enabled = stream_requested and not sort_by and not limit

    all_tracks = library.searchTracks()
    total_tracks = len(all_tracks)
    log.info(f"Fetched {total_tracks} tracks from {config.get('library', LIBRARY_NAME)}")

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

    def flush_stream_buffer():
        nonlocal playlist_obj, stream_buffer, deleted_existing
        if not stream_buffer:
            return
        if existing and not deleted_existing:
            try:
                existing.delete()
            except Exception as exc:
                log.warning(f"Failed to delete existing playlist '{name}': {exc}")
            deleted_existing = True
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
        log.info(f"✅ Finished building '{name}' ({match_count} tracks)")
        return

    if resolved_sort_by:
        sort_desc = config.get("sort_desc", True)
        sort_value_cache = {}

        def _get_sort_value(track):
            cache_key = getattr(track, "ratingKey", None)
            if cache_key in sort_value_cache:
                return sort_value_cache[cache_key]

            if resolved_sort_by == "spotifyPopularity":
                popularity_source = None
                popularity_value = None

                if allmusic_provider and allmusic_provider.is_enabled:
                    popularity_value = allmusic_provider.get_popularity(
                        track,
                        spotify_provider=spotify_provider,
                        playlist_logger=log,
                    )
                    if popularity_value is not None:
                        popularity_source = "allmusic"

                if (popularity_value is None) and spotify_provider and spotify_provider.is_enabled:
                    if debug_logging:
                        log.debug(
                            "Falling back to Spotify popularity for track '%s'",
                            getattr(track, "title", "<unknown>"),
                        )
                    popularity_value = spotify_provider.get_popularity(track)
                    if popularity_value is not None:
                        popularity_source = "spotify"

                if popularity_value is not None:
                    try:
                        popularity_value = float(popularity_value)
                    except (TypeError, ValueError):
                        popularity_value = None

                if debug_logging:
                    log.debug(
                        "Popularity source for track '%s': %s (value=%s)",
                        getattr(track, "title", "<unknown>"),
                        popularity_source or "none",
                        popularity_value,
                    )

                if popularity_value is None:
                    sentinel = float("-inf") if sort_desc else float("inf")
                    sort_value_cache[cache_key] = sentinel
                else:
                    sort_value_cache[cache_key] = popularity_value
                return sort_value_cache[cache_key]

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

                canonical_value = get_canonical_field_for_track(track, resolved_sort_by)
                canonical_numeric = _coerce_numeric(canonical_value)

                track_album_guid = getattr(track, "parentGuid", None) or _get_album_guid(track)
                canonical_album_guid = None

                if canonical_numeric is not None:
                    canonical_album_guid = get_canonical_field_for_track(track, "parentGuid")

                normalized_track_guid = _normalize_plex_guid(track_album_guid)
                normalized_canonical_guid = _normalize_plex_guid(canonical_album_guid)

                is_special = False
                special_reasons = []

                if (
                    normalized_track_guid
                    and normalized_canonical_guid
                    and normalized_track_guid != normalized_canonical_guid
                ):
                    is_special = True
                    special_reasons.append("album GUID mismatch")

                if canonical_numeric is not None:
                    if direct_numeric is None:
                        is_special = True
                        special_reasons.append("missing direct value")
                    elif direct_numeric == 0 and canonical_numeric > 0:
                        is_special = True
                        special_reasons.append("direct zero but canonical > 0")

                if debug_logging:
                    log.debug(
                        "Direct %s=%s (guid=%s); canonical %s=%s (guid=%s)",
                        resolved_sort_by,
                        direct_numeric,
                        normalized_track_guid,
                        resolved_sort_by,
                        canonical_numeric,
                        normalized_canonical_guid,
                    )

                if is_special:
                    if debug_logging:
                        log.debug(
                            "Identified potential special edition (reasons: %s)",
                            "; ".join(special_reasons) if special_reasons else "none",
                        )
                    chosen_value = canonical_numeric if canonical_numeric is not None else direct_numeric
                else:
                    chosen_value = direct_numeric if direct_numeric is not None else canonical_numeric

                if debug_logging:
                    log.debug(
                        "Chosen %s=%s for track '%s'",
                        resolved_sort_by,
                        chosen_value,
                        getattr(track, "title", "<unknown>"),
                    )

                sort_value_cache[cache_key] = chosen_value
                return chosen_value

            if resolved_sort_by in CANONICAL_ONLY_FIELDS:
                canonical_value = get_canonical_field_for_track(track, resolved_sort_by)
                if canonical_value is None:
                    sort_value_cache[cache_key] = None
                    return None
                value = canonical_value
            else:
                value = getattr(track, resolved_sort_by, None)
                if resolved_sort_by in CANONICAL_SORT_FIELDS:
                    canonical_value = get_canonical_field_for_track(track, resolved_sort_by)
                    if canonical_value is not None:
                        value = canonical_value
            if value is not None:
                if (
                    resolved_sort_by in CANONICAL_NUMERIC_FIELDS
                    and isinstance(value, str)
                ):
                    stripped_value = value.strip()
                    if stripped_value:
                        try:
                            value = float(stripped_value)
                        except ValueError:
                            pass
                sort_value_cache[cache_key] = value
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
                sort_value_cache[cache_key] = None
                return None

            xml_value = parse_field_from_xml(xml_text, resolved_sort_by)

            if isinstance(xml_value, (list, set, tuple)):
                xml_value = next(iter(xml_value), None)

            if isinstance(xml_value, str):
                stripped = xml_value.strip()
                if stripped:
                    try:
                        sort_value_cache[cache_key] = float(stripped)
                        return sort_value_cache[cache_key]
                    except ValueError:
                        sort_value_cache[cache_key] = stripped
                        return sort_value_cache[cache_key]
                sort_value_cache[cache_key] = None
                return None

            if isinstance(xml_value, (int, float)):
                sort_value_cache[cache_key] = float(xml_value)
                return sort_value_cache[cache_key]

            sort_value_cache[cache_key] = xml_value
            return xml_value

        def _normalize_sort_value(value):
            """Return a comparable representation for sorting, handling mixed types safely."""
            if value is None:
                return None
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

        matched_tracks.sort(key=cmp_to_key(_compare_tracks))

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
        log.info(f"✅ Finished building '{name}' (0 tracks)")
        return

    for chunk in chunked(matched_tracks, chunk_size):
        try:
            if playlist_obj is None:
                playlist_obj = plex.createPlaylist(name, items=chunk)
            else:
                playlist_obj.addItems(chunk)
        except Exception as exc:
            log.error(f"Failed to update playlist '{name}': {exc}")
            raise

    apply_playlist_cover(playlist_obj, cover_path)
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
        AllMusicPopularityProvider.save_shared_cache()
        if playlist_handler:
            log.debug(
                "Closing per-playlist debug logging for '%s'",
                name,
            )
            playlist_handler.close()

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
def run_all_playlists():
    if not playlists_data:
        logger.warning("No playlists defined. Nothing to process.")
        return

    errors = False
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_playlist, name, cfg): name for name, cfg in playlists_data.items()}
        for future in as_completed(futures):
            playlist_name = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors = True
                logger.exception(f"Playlist '{playlist_name}' failed: {exc}")

    if errors:
        raise RuntimeError("One or more playlists failed to build.")

    logger.info("✅ All playlists processed successfully.")

if __name__ == "__main__":
    if CACHE_ONLY:
        build_metadata_cache()
        logger.info("Cache-only mode complete. Exiting.")
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
