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
raw_playlists = load_yaml("/app/playlists.yml")
playlists_data = raw_playlists.get("playlists", {}) if isinstance(raw_playlists, dict) else {}

# ----------------------------
# Metadata Cache
# ----------------------------
CACHE_FILE = "/app/metadata_cache.json"

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


def chunked(iterable, size):
    """Yield fixed-size chunks from a list."""
    if size <= 0:
        size = 1
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]

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
        # Plex responses wrap metadata inside a <MediaContainer> root
        node = root.find("./Directory") or root.find("./Track") or root.find("./Video") or root.find("./Photo")
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

    # Helper to extract and normalize field values
    def extract_values(source_key):
        if not source_key:
            return []
        try:
            xml_text = fetch_full_metadata(source_key)
            xml_val = parse_field_from_xml(xml_text, field)
            if xml_val:
                # normalize to string list
                return [str(v).strip() for v in xml_val] if isinstance(xml_val, (list, set)) else [str(xml_val).strip()]
        except Exception as e:
            logger.debug(f"Metadata error for key {source_key}: {e}")
        return []

    # 1️⃣ Try direct object field (most accurate)
    val = getattr(track, field, None)
    if val:
        if isinstance(val, list):
            values.update(str(v).strip() for v in val)
        else:
            values.add(str(val).strip())

    # 2️⃣ Try cached XML for the track
    values.update(extract_values(track.ratingKey))

    # 3️⃣ Try album level
    if hasattr(track, "parentRatingKey"):
        values.update(extract_values(track.parentRatingKey))

    # 4️⃣ Try artist level
    if hasattr(track, "grandparentRatingKey"):
        values.update(extract_values(track.grandparentRatingKey))

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

        if playlist_handler:
            logger.addHandler(playlist_handler)
            logger.debug(
                "Per-playlist debug logging for '%s' → %s",
                name,
                playlist_log_path,
            )

        logger.info(f"Building playlist: {name}")
        filters = config.get("plex_filter", [])
        library = plex.library.section(config.get("library", LIBRARY_NAME))
        limit = config.get("limit")
        sort_by = config.get("sort_by")
        sort_field_aliases = {
            "popularity": "ratingCount",
        }
        resolved_sort_by = sort_field_aliases.get(sort_by, sort_by)
        if sort_by and sort_by != resolved_sort_by:
            logger.debug(
                "Sort field '%s' mapped to Plex field '%s'",
                sort_by,
                resolved_sort_by,
            )
        chunk_size = config.get("chunk_size", PLAYLIST_CHUNK_SIZE)
        stream_requested = config.get("stream_while_filtering", False)
        stream_enabled = stream_requested and not sort_by and not limit

        all_tracks = library.searchTracks()
        total_tracks = len(all_tracks)
        logger.info(f"Fetched {total_tracks} tracks from {config.get('library', LIBRARY_NAME)}")

        try:
            existing = plex.playlist(name)
        except Exception:
            existing = None
        deleted_existing = False

        matched_tracks = []
        stream_buffer = []
        playlist_obj = None
        match_count = 0
        debug_logging = logger.isEnabledFor(logging.DEBUG)

        def flush_stream_buffer():
            nonlocal playlist_obj, stream_buffer, deleted_existing
            if not stream_buffer:
                return
            if existing and not deleted_existing:
                try:
                    existing.delete()
                except Exception as exc:
                    logger.warning(f"Failed to delete existing playlist '{name}': {exc}")
                deleted_existing = True
            try:
                if playlist_obj is None:
                    playlist_obj = plex.createPlaylist(name, items=list(stream_buffer))
                else:
                    playlist_obj.addItems(list(stream_buffer))
            except Exception as exc:
                logger.error(f"Failed to update playlist '{name}': {exc}")
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
                    logger.debug(
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
                        logger.debug(
                            "  Condition: field='%s', operator='%s', expected=%s, match_all=%s",
                            field,
                            operator,
                            expected,
                            match_all
                        )
                        logger.debug("    Extracted value: %s", val)
                    if not check_condition(val, operator, expected, match_all):
                        if debug_logging:
                            logger.debug("    ❌ Condition failed for field '%s'", field)
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
                        logger.debug("    ✅ Track matched all conditions")
                pbar.update(1)

        if stream_enabled:
            flush_stream_buffer()
            if not deleted_existing and existing:
                try:
                    existing.delete()
                except Exception as exc:
                    logger.warning(f"Failed to delete existing playlist '{name}': {exc}")
                deleted_existing = True
            logger.info(f"Playlist '{name}' → {match_count} matching tracks")
            if match_count == 0:
                logger.info(f"No tracks matched for '{name}'. Playlist will not be recreated.")
            logger.info(f"✅ Finished building '{name}' ({match_count} tracks)")
            return

        if resolved_sort_by:
            sort_desc = config.get("sort_desc", True)
            sort_value_cache = {}

            def _get_sort_value(track):
                cache_key = getattr(track, "ratingKey", None)
                if cache_key in sort_value_cache:
                    return sort_value_cache[cache_key]

                value = getattr(track, resolved_sort_by, None)
                if value is not None:
                    sort_value_cache[cache_key] = value
                    return value

                try:
                    xml_text = fetch_full_metadata(track.ratingKey)
                except Exception as exc:
                    logger.debug(
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

        logger.info(f"Playlist '{name}' → {match_count} matching tracks")

        if existing and not deleted_existing:
            try:
                existing.delete()
            except Exception as exc:
                logger.warning(f"Failed to delete existing playlist '{name}': {exc}")

        if not matched_tracks:
            logger.info(f"No tracks matched for '{name}'. Playlist will not be recreated.")
            logger.info(f"✅ Finished building '{name}' (0 tracks)")
            return

        for chunk in chunked(matched_tracks, chunk_size):
            try:
                if playlist_obj is None:
                    playlist_obj = plex.createPlaylist(name, items=chunk)
                else:
                    playlist_obj.addItems(chunk)
            except Exception as exc:
                logger.error(f"Failed to update playlist '{name}': {exc}")
                raise

        logger.info(f"✅ Finished building '{name}' ({match_count} tracks)")
    finally:
        if playlist_handler:
            logger.debug(
                "Closing per-playlist debug logging for '%s'",
                name,
            )
            logger.removeHandler(playlist_handler)
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
