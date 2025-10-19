import os
import re
import yaml
import logging
import requests
import json
import time
import threading
from plexapi.server import PlexServer
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cmp_to_key
from xml.etree import ElementTree as ET
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
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.debug("Canonical search XML parse error.")
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
    if not album:
        return None

    if CACHE_ONLY:
        return None

    album_query_candidates = []
    for candidate in (album, _strip_parenthetical(album)):
        candidate = (candidate or "").strip()
        if candidate and candidate not in album_query_candidates:
            album_query_candidates.append(candidate)

    query_strings = []
    for base in album_query_candidates or [album]:
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

    if not deduped_queries:
        deduped_queries = [album]

    album_norms = _collect_normalized_candidates(album_query_candidates or [album])
    artist_norms = _collect_normalized_candidates([artist])
    year_token = _extract_year_token(year)

    search_endpoints = [
        f"{METADATA_PROVIDER_URL}/library/metadata/search",
        f"{METADATA_PROVIDER_URL}/library/search",
    ]

    for query in deduped_queries:
        for endpoint in search_endpoints:
            params = {
                "query": query,
                "type": 9,
                "X-Plex-Token": PLEX_TOKEN,
            }

            try:
                response = requests.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.debug(
                    "Canonical metadata search failed for query '%s' via %s: %s",
                    query,
                    endpoint,
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
                logger.debug(
                    "Canonical metadata search resolved '%s' by '%s' (%s) to GUID %s",
                    album,
                    artist,
                    year,
                    _normalize_plex_guid(guid),
                )
                return guid

    logger.debug(
        "Canonical metadata search could not resolve '%s' by '%s' (%s)",
        album,
        artist,
        year,
    )
    return None


def _lookup_canonical_guid_for_track(track):
    artist = getattr(track, "grandparentTitle", None)
    album = getattr(track, "parentTitle", None)
    year = getattr(track, "parentYear", None)

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
        sort_field_aliases = {
            "popularity": "ratingCount",
        }
        resolved_sort_by = sort_field_aliases.get(sort_by, sort_by)
        if sort_by and sort_by != resolved_sort_by:
            log.debug(
                "Sort field '%s' mapped to Plex field '%s'",
                sort_by,
                resolved_sort_by,
            )
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
    finally:
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
