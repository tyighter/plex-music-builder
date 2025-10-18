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
print("Fetching sample tracks...\n")

# ─────────────────────────────────────────────
# Print field data for a few sample tracks
# ─────────────────────────────────────────────
for t in library.searchTracks(limit=10):
    artist = t.artist().title if callable(t.artist) else t.artist
    print({
        "title": t.title,
        "artist": artist,
        "album": t.album().title if hasattr(t, "album") else None,
        "rating": getattr(t, "userRating", None),
        "audienceRating": getattr(t, "audienceRating", None),
        "ratingCount": getattr(t, "ratingCount", None),
        "viewCount": getattr(t, "viewCount", None),
        "guid": getattr(t, "guid", None),
    })
