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
print("Fetching Stop Making Sense albums...\n")

TITLE_MATCH = "stop making sense"
ARTIST_NAME = "talking heads"


def find_stop_making_sense_albums(section):
    candidates = section.searchAlbums(title="Stop Making Sense")
    sms_albums = []
    for album in candidates:
        artist = album.artist().title if callable(album.artist) else album.artist
        title = album.title or ""

        if not title or TITLE_MATCH not in title.lower():
            continue

        if artist and ARTIST_NAME in artist.lower():
            sms_albums.append(album)

    if not sms_albums:
        raise RuntimeError("Could not locate any Stop Making Sense albums by Talking Heads in Plex.")

    return sms_albums


albums = find_stop_making_sense_albums(library)
print(f"Found {len(albums)} Stop Making Sense album(s) by {ARTIST_NAME.title()}\n")


def dump_track_metadata(track):
    element = track._server.query(track.key)
    from xml.etree import ElementTree as ET

    xml_text = ET.tostring(element, encoding="unicode")
    print(f"=== Track: {track.title} (Disc {track.parentIndex} • Track {track.index}) ===")
    print(xml_text)
    print()
for album in albums:
    artist_name = album.artist().title if callable(album.artist) else album.artist
    tracks = album.tracks()

    print(f"=== Album: {album.title} by {artist_name} ===")
    print(f"Total tracks: {len(tracks)}\n")

    for track in tracks:
        dump_track_metadata(track)
