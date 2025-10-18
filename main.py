import os
import yaml
import logging
import requests
import json
import time
from plexapi.server import PlexServer
from concurrent.futures import ThreadPoolExecutor, as_completed
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

LOG_LEVEL = cfg.get("logging", {}).get("level", "INFO").upper()

if not PLEX_URL or not PLEX_TOKEN:
    raise EnvironmentError("PLEX_URL and PLEX_TOKEN must be set in config.yml")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

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
    logger.info(f"Building playlist: {name}")
    filters = config.get("plex_filter", [])
    library = plex.library.section(config.get("library", LIBRARY_NAME))
    limit = config.get("limit")
    sort_by = config.get("sort_by")

    all_tracks = library.searchTracks()
    total_tracks = len(all_tracks)
    logger.info(f"Fetched {total_tracks} tracks from {config.get('library', LIBRARY_NAME)}")

    matched_tracks = []

    with tqdm(total=total_tracks, desc=f"Filtering '{name}'", unit="track", dynamic_ncols=True) as pbar:
        for track in all_tracks:
            keep = True
            for f in filters:
                field = f["field"]
                operator = f["operator"]
                expected = f["value"]
                match_all = f.get("match_all", True)

                val = get_field_value(track, field)
                if not check_condition(val, operator, expected, match_all):
                    keep = False
                    break
            if keep:
                matched_tracks.append(track)
            pbar.update(1)

    if sort_by:
        matched_tracks.sort(key=lambda t: getattr(t, sort_by, None), reverse=True)
    if limit:
        matched_tracks = matched_tracks[:limit]

    logger.info(f"Playlist '{name}' → {len(matched_tracks)} matching tracks")

    try:
        existing = plex.playlist(name)
        if existing:
            existing.delete()
    except Exception:
        pass

    if matched_tracks:
        plex.createPlaylist(name, items=matched_tracks)

    logger.info(f"✅ Finished building '{name}' ({len(matched_tracks)} tracks)")

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
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_playlist, name, cfg) for name, cfg in playlists_data.items()]
        for f in as_completed(futures):
            f.result()
    logger.info("✅ All playlists processed successfully.")

if __name__ == "__main__":
    if CACHE_ONLY:
        build_metadata_cache()
        logger.info("Cache-only mode complete. Exiting.")
    elif RUN_FOREVER:
        logger.info("Running in loop mode.")
        while True:
            run_all_playlists()
            logger.info(f"Sleeping for {REFRESH_INTERVAL} minutes before next run...")
            time.sleep(REFRESH_INTERVAL * 60)
    else:
        run_all_playlists()
