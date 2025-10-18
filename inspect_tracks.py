import yaml
from plexapi.server import PlexServer

# ─────────────────────────────────────────────
# Load Plex connection details
# ─────────────────────────────────────────────
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_yaml("config.yml")


def get_config_value(section, *candidate_keys, required=True):
    """Return the first matching key in *candidate_keys* with a truthy value."""

    for key in candidate_keys:
        if key in section and section[key]:
            return section[key]

    if required:
        tried = ", ".join(candidate_keys)
        available = ", ".join(section.keys()) if hasattr(section, "keys") else "<none>"
        raise KeyError(
            f"Missing configuration value. Tried keys [{tried}] within section containing [{available}]."
        )

    return None


plex_cfg = cfg.get("plex", {})

base_url = get_config_value(plex_cfg, "base_url", "PLEX_URL")
token = get_config_value(plex_cfg, "token", "PLEX_TOKEN")
library_name = get_config_value(plex_cfg, "library_name")

plex = PlexServer(base_url, token)
library = plex.library.section(library_name)

print(f"Connected to Plex: {plex.friendlyName}")
print(f"Using library: {library_name}")
print("Fetching Revolver (Super Deluxe) tracks...\n")

ALBUM_TITLE_KEYWORDS = ["revolver", "super deluxe"]
ARTIST_NAME = "The Beatles"


def find_special_edition_album(section):
    candidates = section.searchAlbums(title="Revolver", artist=ARTIST_NAME)
    for album in candidates:
        artist = album.artist().title if callable(album.artist) else album.artist
        title = album.title
        if not artist or ARTIST_NAME.lower() not in artist.lower():
            continue

        lower_title = title.lower()
        if all(keyword in lower_title for keyword in ALBUM_TITLE_KEYWORDS):
            return album

    raise RuntimeError("Could not locate Revolver (Super Deluxe) by The Beatles in Plex.")


album = find_special_edition_album(library)
tracks = album.tracks()
artist_name = album.artist().title if callable(album.artist) else album.artist
print(f"Found album: {album.title} by {artist_name}")
print(f"Total tracks: {len(tracks)}\n")


def dump_track_metadata(track):
    element = track._server.query(track.key)
    from xml.etree import ElementTree as ET

    xml_text = ET.tostring(element, encoding="unicode")
    print(f"=== Track: {track.title} (Disc {track.parentIndex} • Track {track.index}) ===")
    print(xml_text)
    print()


for track in tracks:
    dump_track_metadata(track)
