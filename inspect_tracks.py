import yaml
from plexapi.server import PlexServer

# ─────────────────────────────────────────────
# Load Plex connection details
# ─────────────────────────────────────────────
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_yaml("config.yml")

base_url = cfg["plex"]["base_url"]
token = cfg["plex"]["token"]
library_name = cfg["plex"]["library_name"]

plex = PlexServer(base_url, token)
library = plex.library.section(library_name)

print(f"Connected to Plex: {plex.friendlyName}")
print(f"Using library: {library_name}")
print("Fetching Revolver (Super Deluxe) tracks...\n")

ALBUM_TITLE_KEYWORDS = ["revolver", "super deluxe"]
ARTIST_NAME = "the beatles"


def find_special_edition_album(section):
    candidates = section.searchAlbums("Revolver")
    for album in candidates:
        artist = album.artist().title if callable(album.artist) else album.artist
        title = album.title
        if not artist or ARTIST_NAME not in artist.lower():
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
