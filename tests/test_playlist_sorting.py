import logging
from collections import OrderedDict

import yaml


def write_yaml(path, data):
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


class DummyTqdm:
    def __init__(self, *_, **__):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, *_):  # pragma: no cover - trivial helper
        return None


class _StubPlaylist:
    def __init__(self, name, initial_items):
        self.title = name
        self.items = list(initial_items)

    def addItems(self, items):
        self.items.extend(items)

    def uploadPoster(self, filepath=None):  # pragma: no cover - stub
        pass


class _StubSection:
    def __init__(self, tracks):
        self._tracks = list(tracks)

    def searchTracks(self):
        return list(self._tracks)


class _StubLibrary:
    def __init__(self, tracks):
        self._section = _StubSection(tracks)

    def section(self, name):
        return self._section


class _StubServer:
    def __init__(self, tracks):
        self.library = _StubLibrary(tracks)
        self.created_playlist = None

    def playlist(self, name):
        raise Exception("not found")

    def createPlaylist(self, name, items):
        playlist = _StubPlaylist(name, items)
        self.created_playlist = playlist
        return playlist


def _prepare_playlist_build(monkeypatch, tracks):
    import main

    server = _StubServer(tracks)
    monkeypatch.setattr(main, "get_plex_server", lambda: server)
    monkeypatch.setattr(main, "apply_playlist_cover", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "tqdm", DummyTqdm)
    return main, server


def test_save_playlists_alphabetizes_entries(tmp_path, monkeypatch):
    from gui import save_playlists

    playlist_path = tmp_path / "playlists.yml"
    monkeypatch.setattr("gui.PLAYLISTS_PATH", playlist_path)

    payload = {
        "defaults": {},
        "playlists": [
            {"name": "Rock", "limit": 0, "artist_limit": 0, "album_limit": 0},
            {"name": "acoustic", "limit": 0, "artist_limit": 0, "album_limit": 0},
            {"name": "Blues", "limit": 0, "artist_limit": 0, "album_limit": 0},
        ],
    }

    save_playlists(payload)

    with playlist_path.open("r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert list(saved["playlists"].keys()) == ["acoustic", "Blues", "Rock"]


def test_save_single_playlist_preserves_sorted_order(tmp_path, monkeypatch):
    from gui import save_single_playlist

    playlist_path = tmp_path / "playlists.yml"
    monkeypatch.setattr("gui.PLAYLISTS_PATH", playlist_path)

    initial_data = {
        "defaults": {},
        "playlists": OrderedDict(
            [
                ("Rock", {"limit": 0, "artist_limit": 0, "album_limit": 0}),
                ("acoustic", {"limit": 0, "artist_limit": 0, "album_limit": 0}),
            ]
        ),
    }
    write_yaml(playlist_path, initial_data)

    save_single_playlist({"name": "Blues", "limit": 0, "artist_limit": 0, "album_limit": 0})

    with playlist_path.open("r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert list(saved["playlists"].keys()) == ["acoustic", "Blues", "Rock"]


def test_load_playlists_returns_sorted_entries(tmp_path, monkeypatch):
    from gui import load_playlists

    playlist_path = tmp_path / "playlists.yml"
    monkeypatch.setattr("gui.PLAYLISTS_PATH", playlist_path)

    data = {
        "defaults": {},
        "playlists": {
            "Rock": {"limit": 0},
            "acoustic": {"limit": 0},
            "Blues": {"limit": 0},
        },
    }

    write_yaml(playlist_path, data)

    result = load_playlists()
    names = [playlist["name"] for playlist in result["playlists"]]

    assert names == ["acoustic", "Blues", "Rock"]


def test_load_playlists_includes_after_sort(tmp_path, monkeypatch):
    from gui import load_playlists

    playlist_path = tmp_path / "playlists.yml"
    monkeypatch.setattr("gui.PLAYLISTS_PATH", playlist_path)

    data = {
        "defaults": {},
        "playlists": {"Mix": {"limit": 0, "after_sort": "alphabetical"}},
    }

    write_yaml(playlist_path, data)

    result = load_playlists()
    entry = next(playlist for playlist in result["playlists"] if playlist["name"] == "Mix")

    assert entry["after_sort"] == "alphabetical"


def test_save_playlists_persists_default_popularity_boosts(tmp_path, monkeypatch):
    from gui import save_playlists

    playlist_path = tmp_path / "playlists.yml"
    monkeypatch.setattr("gui.PLAYLISTS_PATH", playlist_path)

    payload = {
        "defaults": {
            "popularity_boosts": [
                {
                    "conditions": [
                        {
                            "field": "genres",
                            "operator": "equals",
                            "value": "Rock",
                            "match_all": True,
                        },
                        {
                            "field": "year",
                            "operator": "greater_than",
                            "value": "1980",
                        },
                    ],
                    "boost": 1.5,
                },
                {
                    "conditions": [
                        {
                            "field": "moods",
                            "operator": "equals",
                            "value": "Energetic, Happy",
                            "match_all": False,
                        }
                    ],
                    "boost": 2,
                },
            ]
        },
        "playlists": [],
    }

    save_playlists(payload)

    with playlist_path.open("r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["defaults"]["popularity_boosts"] == [
        {
            "conditions": [
                {"field": "genres", "operator": "equals", "value": "Rock"},
                {"field": "year", "operator": "greater_than", "value": 1980},
            ],
            "boost": 1.5,
        },
        {
            "conditions": [
                {
                    "field": "moods",
                    "operator": "equals",
                    "value": ["Energetic", "Happy"],
                    "match_all": False,
                }
            ],
            "boost": 2.0,
        },
    ]


def test_save_playlists_persists_playlist_popularity_boosts(tmp_path, monkeypatch):
    from gui import save_playlists

    playlist_path = tmp_path / "playlists.yml"
    monkeypatch.setattr("gui.PLAYLISTS_PATH", playlist_path)

    payload = {
        "defaults": {},
        "playlists": [
            {
                "name": "Boosted",
                "limit": 0,
                "artist_limit": 0,
                "album_limit": 0,
                "popularity_boosts": [
                    {
                        "conditions": [
                            {
                                "field": "album",
                                "operator": "contains",
                                "value": "Mix",
                            }
                        ],
                        "boost": 3,
                    }
                ],
            }
        ],
    }

    save_playlists(payload)

    with playlist_path.open("r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    playlist_config = saved["playlists"]["Boosted"]

    assert playlist_config["popularity_boosts"] == [
        {
            "conditions": [
                {"field": "album", "operator": "contains", "value": "Mix"},
            ],
            "boost": 3.0,
        }
    ]


def test_save_playlists_persists_after_sort(tmp_path, monkeypatch):
    from gui import save_playlists

    playlist_path = tmp_path / "playlists.yml"
    monkeypatch.setattr("gui.PLAYLISTS_PATH", playlist_path)

    payload = {
        "defaults": {},
        "playlists": [
            {
                "name": "Chill",
                "limit": 25,
                "artist_limit": 0,
                "album_limit": 0,
                "after_sort": "alphabetical",
            }
        ],
    }

    save_playlists(payload)

    with playlist_path.open("r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["playlists"]["Chill"]["after_sort"] == "alphabetical"


def test_save_single_playlist_persists_after_sort(tmp_path, monkeypatch):
    from gui import save_single_playlist

    playlist_path = tmp_path / "playlists.yml"
    monkeypatch.setattr("gui.PLAYLISTS_PATH", playlist_path)

    payload = {
        "name": "Focus",
        "limit": 100,
        "artist_limit": 0,
        "album_limit": 0,
        "after_sort": "alphabetical",
    }

    save_single_playlist(payload)

    with playlist_path.open("r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["playlists"]["Focus"]["after_sort"] == "alphabetical"


def test_sort_tracks_in_place_alphabetical():
    import main

    class Track:
        def __init__(self, rating_key, title, artist):
            self.ratingKey = rating_key
            self.title = title
            self.grandparentTitle = artist

    tracks = [
        Track("1", "Beta", "Artist"),
        Track("2", "Alpha", "Artist"),
        Track("3", "Gamma", "Artist"),
    ]

    duration = main._sort_tracks_in_place(
        tracks,
        "__alphabetical__",
        False,
        logging.getLogger("test"),
        {},
        {},
        {},
        debug_logging=False,
    )

    assert [track.title for track in tracks] == ["Alpha", "Beta", "Gamma"]
    assert duration >= 0.0


def test_run_playlist_build_after_sort_case_insensitive(monkeypatch):
    class StubTrack:
        def __init__(self, rating_key, title, artist, popularity):
            self.ratingKey = rating_key
            self.title = title
            self.grandparentTitle = artist
            self.ratingCount = popularity

    tracks = [
        StubTrack("1", "Gamma", "Artist", 30),
        StubTrack("2", "Alpha", "Artist", 10),
        StubTrack("3", "Beta", "Artist", 40),
    ]

    main, server = _prepare_playlist_build(monkeypatch, tracks)

    config = {
        "limit": 3,
        "sort_by": "popularity",
        "after_sort": "Alphabetical",
    }

    logger = logging.getLogger("test_after_sort")
    logger.setLevel(logging.INFO)

    main._run_playlist_build("Test", config, logger, None, None)

    assert server.created_playlist is not None
    titles = [track.title for track in server.created_playlist.items]
    assert titles == ["Alpha", "Beta", "Gamma"]


def test_run_playlist_build_limit_applies_before_after_sort(monkeypatch):
    class StubTrack:
        def __init__(self, rating_key, title, popularity):
            self.ratingKey = rating_key
            self.title = title
            self.grandparentTitle = "Artist"
            self.parentTitle = f"Album {title}"
            self.ratingCount = popularity

    tracks = [
        StubTrack("1", "Alpha", 10),
        StubTrack("2", "Beta", 50),
        StubTrack("3", "Gamma", 40),
    ]

    main, server = _prepare_playlist_build(monkeypatch, tracks)

    config = {
        "limit": 2,
        "sort_by": "popularity",
        "after_sort": "alphabetical",
    }

    logger = logging.getLogger("test_limit_before_after_sort")
    logger.setLevel(logging.INFO)

    main._run_playlist_build("Test", config, logger, None, None)

    assert server.created_playlist is not None
    titles = [track.title for track in server.created_playlist.items]
    assert titles == ["Beta", "Gamma"]


def test_run_playlist_build_sort_by_oldest_uses_album_year(monkeypatch):
    class StubTrack:
        def __init__(
            self,
            rating_key,
            title,
            parent_year,
            track_year,
            track_date,
            parent_date,
        ):
            self.ratingKey = rating_key
            self.title = title
            self.grandparentTitle = "Artist"
            self.parentTitle = f"Album {title}"
            self.parentYear = parent_year
            self.year = track_year
            self.originallyAvailableAt = track_date
            self.parentOriginallyAvailableAt = parent_date

    tracks = [
        StubTrack("1", "Modern", 2020, 1970, "1970-01-01", "2020-06-01"),
        StubTrack("2", "Classic", 1980, 2022, "2022-02-02", "1980-09-09"),
        StubTrack("3", "Recent", 2010, 2010, "2010-03-03", "2010-07-07"),
    ]

    main, server = _prepare_playlist_build(monkeypatch, tracks)
    main._ALBUM_YEAR_CACHE.clear()
    main._ALBUM_YEAR_MISS_KEYS.clear()

    config = {
        "sort_by": "oldest_first",
    }

    logger = logging.getLogger("test_sort_by_album_year")
    logger.setLevel(logging.INFO)

    main._run_playlist_build("Test", config, logger, None, None)

    assert server.created_playlist is not None
    titles = [track.title for track in server.created_playlist.items]
    assert titles == ["Classic", "Recent", "Modern"]


def test_run_playlist_build_after_sort_newest_uses_album_year(monkeypatch):
    class StubTrack:
        def __init__(self, rating_key, title, parent_year, track_year):
            self.ratingKey = rating_key
            self.title = title
            self.grandparentTitle = "Artist"
            self.parentTitle = f"Album {title}"
            self.parentYear = parent_year
            self.year = track_year
            self.originallyAvailableAt = f"{track_year}-01-01"
            self.parentOriginallyAvailableAt = f"{parent_year}-06-01"

    tracks = [
        StubTrack("1", "Vintage", 1985, 2020),
        StubTrack("2", "Modern", 2015, 1995),
        StubTrack("3", "Contemporary", 2005, 2005),
    ]

    main, server = _prepare_playlist_build(monkeypatch, tracks)
    main._ALBUM_YEAR_CACHE.clear()
    main._ALBUM_YEAR_MISS_KEYS.clear()

    config = {
        "sort_by": "alphabetical",
        "after_sort": "newest_first",
    }

    logger = logging.getLogger("test_after_sort_album_year")
    logger.setLevel(logging.INFO)

    main._run_playlist_build("Test", config, logger, None, None)

    assert server.created_playlist is not None
    titles = [track.title for track in server.created_playlist.items]
    assert titles == ["Modern", "Contemporary", "Vintage"]
