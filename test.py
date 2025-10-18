from plexapi.server import PlexServer

PLEX_URL = "http://192.168.111.77:32400"
PLEX_TOKEN = "isFgSyj8THjGwnitrzdP"

plex = PlexServer(PLEX_URL, PLEX_TOKEN)
library = plex.library.section("Music")
track = library.searchTracks(title="Jeopardy")[0]

# Assuming get_field_value() is defined in your script
from main import get_field_value
print(get_field_value(track, "genres"))
