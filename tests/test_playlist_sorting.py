import datetime
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
        self._items = list(initial_items)
        self.removed_batches = []
        self.deleted = False

    def addItems(self, items):
        self._items.extend(items)

    def removeItems(self, items):
        self.removed_batches.append(list(items))
        to_remove = {getattr(item, "ratingKey", id(item)) for item in items}
        self._items = [
            item for item in self._items if getattr(item, "ratingKey", id(item)) not in to_remove
        ]

    def delete(self):
        self.deleted = True

    def items(self):
        return list(self._items)

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


class _ExistingPlaylistServer(_StubServer):
    def __init__(self, tracks, existing_playlist):
        super().__init__(tracks)
        self._existing_playlist = existing_playlist

    def playlist(self, name):
        return self._existing_playlist


def test_save_playlists_alphabetizes_entries(tmp_path, monkeypatch):
    from gui import save_playlists

    playlist_path = tmp_path / "playlists.yml"
    monkeypatch.setattr("gui.PLAYLISTS_PATH", playlist_path)

    payload = {
        "defaults": {},
        "playlists": [
            {
                "name": "Rock",
                "limit": 0,
                "artist_limit": 0,
                "album_limit": 0,
                "year_limit": 0,
            },
            {
                "name": "acoustic",
                "limit": 0,
                "artist_limit": 0,
                "album_limit": 0,
                "year_limit": 0,
            },
            {
                "name": "Blues",
                "limit": 0,
                "artist_limit": 0,
                "album_limit": 0,
                "year_limit": 0,
            },
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
                (
                    "Rock",
                    {
                        "limit": 0,
                        "artist_limit": 0,
                        "album_limit": 0,
                        "year_limit": 0,
                    },
                ),
                (
                    "acoustic",
                    {
                        "limit": 0,
                        "artist_limit": 0,
                        "album_limit": 0,
                        "year_limit": 0,
                    },
                ),
            ]
        ),
    }
    write_yaml(playlist_path, initial_data)

    save_single_playlist(
        {
            "name": "Blues",
            "limit": 0,
            "artist_limit": 0,
            "album_limit": 0,
            "year_limit": 0,
        }
    )

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


def test_save_playlists_persists_playlist_auto_build_flag(tmp_path, monkeypatch):
    from gui import save_playlists

    playlist_path = tmp_path / "playlists.yml"
    monkeypatch.setattr("gui.PLAYLISTS_PATH", playlist_path)

    payload = {
        "defaults": {},
        "playlists": [
            {
                "name": "Manual Only",
                "limit": 25,
                "artist_limit": 0,
                "album_limit": 0,
                "year_limit": 0,
                "auto_build": False,
            }
        ],
    }

    save_playlists(payload)

    with playlist_path.open("r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["playlists"]["Manual Only"]["auto_build"] is False


def test_load_playlists_defaults_auto_build_to_true(tmp_path, monkeypatch):
    from gui import load_playlists

    playlist_path = tmp_path / "playlists.yml"
    monkeypatch.setattr("gui.PLAYLISTS_PATH", playlist_path)

    write_yaml(
        playlist_path,
        {
            "defaults": {},
            "playlists": {
                "No Explicit Flag": {"limit": 10},
                "Disabled": {"limit": 10, "auto_build": False},
            },
        },
    )

    result = load_playlists()
    by_name = {entry["name"]: entry for entry in result["playlists"]}

    assert by_name["No Explicit Flag"]["auto_build"] is True
    assert by_name["Disabled"]["auto_build"] is False


def test_get_auto_build_playlist_subset_respects_playlist_override():
    import main

    playlists = {
        "DefaultEnabled": {},
        "ExplicitEnabled": {"auto_build": True},
        "Disabled": {"auto_build": False},
        "StringDisabled": {"auto_build": "no"},
        "StringEnabled": {"auto_build": "yes"},
    }

    selected = main._get_auto_build_playlist_subset(playlists)

    assert set(selected.keys()) == {
        "DefaultEnabled",
        "ExplicitEnabled",
        "StringEnabled",
    }


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
    titles = [track.title for track in server.created_playlist.items()]
    assert titles == ["Alpha", "Beta", "Gamma"]


def test_run_playlist_build_updates_existing_playlist_in_place(monkeypatch):
    import main

    class StubTrack:
        def __init__(self, rating_key, title):
            self.ratingKey = rating_key
            self.title = title
            self.grandparentTitle = "Artist"
            self.parentTitle = f"Album {title}"

    existing_track = StubTrack("legacy", "Legacy")
    replacement_tracks = [
        StubTrack("1", "Alpha"),
        StubTrack("2", "Beta"),
    ]

    existing_playlist = _StubPlaylist("Test", [existing_track])
    server = _ExistingPlaylistServer(replacement_tracks, existing_playlist)
    monkeypatch.setattr(main, "get_plex_server", lambda: server)
    monkeypatch.setattr(main, "apply_playlist_cover", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "tqdm", DummyTqdm)

    logger = logging.getLogger("test_replace_in_place")
    logger.setLevel(logging.INFO)
    main._run_playlist_build("Test", {}, logger, None, None)

    assert server.created_playlist is None
    assert existing_playlist.deleted is False
    assert [track.title for track in existing_playlist.items()] == ["Alpha", "Beta"]


def test_run_playlist_build_falls_back_to_recreate_when_in_place_clear_fails(monkeypatch):
    import main

    class StubTrack:
        def __init__(self, rating_key, title):
            self.ratingKey = rating_key
            self.title = title
            self.grandparentTitle = "Artist"
            self.parentTitle = f"Album {title}"

    class _FailingClearPlaylist(_StubPlaylist):
        def removeItems(self, items):
            raise RuntimeError("cannot clear")

    replacement_tracks = [
        StubTrack("1", "Alpha"),
        StubTrack("2", "Beta"),
    ]

    existing_playlist = _FailingClearPlaylist("Test", [StubTrack("legacy", "Legacy")])
    server = _ExistingPlaylistServer(replacement_tracks, existing_playlist)
    monkeypatch.setattr(main, "get_plex_server", lambda: server)
    monkeypatch.setattr(main, "apply_playlist_cover", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "tqdm", DummyTqdm)

    logger = logging.getLogger("test_fallback_recreate")
    logger.setLevel(logging.INFO)
    main._run_playlist_build("Test", {}, logger, None, None)

    assert existing_playlist.deleted is True
    assert server.created_playlist is not None
    assert [track.title for track in server.created_playlist.items()] == ["Alpha", "Beta"]


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
    titles = [track.title for track in server.created_playlist.items()]
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
    main._ALBUM_RELEASE_DATE_CACHE.clear()
    main._ALBUM_RELEASE_DATE_MISS_KEYS.clear()

    config = {
        "sort_by": "oldest_first",
    }

    logger = logging.getLogger("test_sort_by_album_year")
    logger.setLevel(logging.INFO)

    main._run_playlist_build("Test", config, logger, None, None)

    assert server.created_playlist is not None
    titles = [track.title for track in server.created_playlist.items()]
    assert titles == ["Classic", "Recent", "Modern"]


def test_run_playlist_build_sort_by_oldest_uses_release_date_for_ties(monkeypatch):
    class StubTrack:
        def __init__(
            self,
            rating_key,
            title,
            parent_year,
            parent_release,
        ):
            self.ratingKey = rating_key
            self.title = title
            self.grandparentTitle = "Artist"
            self.parentTitle = f"Album {title}"
            self.parentYear = parent_year
            self.year = parent_year
            self.parentOriginallyAvailableAt = parent_release
            self.originallyAvailableAt = parent_release

    tracks = [
        StubTrack("1", "Late", 2020, "2020-11-11"),
        StubTrack("2", "Early", 2020, "2020-01-01"),
        StubTrack("3", "Middle", 2020, "2020-06-06"),
    ]

    main, server = _prepare_playlist_build(monkeypatch, tracks)
    main._ALBUM_YEAR_CACHE.clear()
    main._ALBUM_YEAR_MISS_KEYS.clear()
    main._ALBUM_RELEASE_DATE_CACHE.clear()
    main._ALBUM_RELEASE_DATE_MISS_KEYS.clear()

    config = {
        "sort_by": "oldest_first",
    }

    logger = logging.getLogger("test_sort_by_album_release_date")
    logger.setLevel(logging.INFO)

    main._run_playlist_build("Test", config, logger, None, None)

    assert server.created_playlist is not None
    titles = [track.title for track in server.created_playlist.items()]
    assert titles == ["Early", "Middle", "Late"]


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
    main._ALBUM_RELEASE_DATE_CACHE.clear()
    main._ALBUM_RELEASE_DATE_MISS_KEYS.clear()

    config = {
        "sort_by": "alphabetical",
        "after_sort": "newest_first",
    }

    logger = logging.getLogger("test_after_sort_album_year")
    logger.setLevel(logging.INFO)

    main._run_playlist_build("Test", config, logger, None, None)

    assert server.created_playlist is not None
    titles = [track.title for track in server.created_playlist.items()]
    assert titles == ["Modern", "Contemporary", "Vintage"]


def test_run_playlist_build_after_sort_newest_uses_release_date_for_ties(monkeypatch):
    class StubTrack:
        def __init__(
            self,
            rating_key,
            title,
            parent_year,
            parent_release,
        ):
            self.ratingKey = rating_key
            self.title = title
            self.grandparentTitle = "Artist"
            self.parentTitle = f"Album {title}"
            self.parentYear = parent_year
            self.year = parent_year
            self.parentOriginallyAvailableAt = parent_release
            self.originallyAvailableAt = parent_release

    tracks = [
        StubTrack("1", "Spring", 2020, "2020-03-03"),
        StubTrack("2", "Winter", 2020, "2020-12-12"),
        StubTrack("3", "Summer", 2020, "2020-07-07"),
    ]

    main, server = _prepare_playlist_build(monkeypatch, tracks)
    main._ALBUM_YEAR_CACHE.clear()
    main._ALBUM_YEAR_MISS_KEYS.clear()
    main._ALBUM_RELEASE_DATE_CACHE.clear()
    main._ALBUM_RELEASE_DATE_MISS_KEYS.clear()

    config = {
        "sort_by": "alphabetical",
        "after_sort": "newest_first",
    }

    logger = logging.getLogger("test_after_sort_release_date")
    logger.setLevel(logging.INFO)

    main._run_playlist_build("Test", config, logger, None, None)

    assert server.created_playlist is not None
    titles = [track.title for track in server.created_playlist.items()]
    assert titles == ["Winter", "Summer", "Spring"]


def test_deduplicate_tracks_prefers_most_popular_by_default():
    import main

    class StubTrack:
        def __init__(self, album, popularity, release_date):
            self.title = "Song"
            self.grandparentTitle = "Artist"
            self.parentTitle = album
            self.ratingCount = popularity
            self.parentRatingCount = popularity
            self.parentOriginallyAvailableAt = release_date
            self.originallyAvailableAt = release_date
            self.guid = "shared-guid"
            self.ratingKey = None

    less_popular = StubTrack("Album B-Side", 10, "2001-01-01")
    more_popular = StubTrack("Album Hit", 50, "2005-05-05")

    log = logging.getLogger("test_deduplicate_popularity")
    log.setLevel(logging.DEBUG)

    deduped, cache, removed, reason_counts = main._deduplicate_tracks(
        [less_popular, more_popular],
        log,
    )

    assert removed == 1
    assert [track.parentTitle for track in deduped] == ["Album Hit"]
    assert cache == {}
    assert reason_counts == {"popularity": 1}


def test_deduplicate_tracks_prefers_oldest_when_configured():
    import main

    class StubTrack:
        def __init__(self, album, popularity, release_date):
            self.title = "Song"
            self.grandparentTitle = "Artist"
            self.parentTitle = album
            self.ratingCount = popularity
            self.parentRatingCount = popularity
            self.parentOriginallyAvailableAt = release_date
            self.originallyAvailableAt = release_date
            self.guid = "shared-guid"
            self.ratingKey = None

    newer = StubTrack("Album Remaster", 80, "2020-06-01")
    older = StubTrack("Album Original", 10, "1980-02-02")

    log = logging.getLogger("test_deduplicate_oldest")
    log.setLevel(logging.DEBUG)

    deduped, _, removed, reason_counts = main._deduplicate_tracks(
        [newer, older],
        log,
        duplicate_tiebreaker="oldest",
    )

    assert removed == 1
    assert [track.parentTitle for track in deduped] == ["Album Original"]
    assert reason_counts == {"oldest": 1}


def test_deduplicate_tracks_prefers_oldest_with_non_padded_dates():
    import main

    class StubTrack:
        def __init__(self, album, popularity, release_date):
            self.title = "Song"
            self.grandparentTitle = "Artist"
            self.parentTitle = album
            self.ratingCount = popularity
            self.parentRatingCount = popularity
            self.parentOriginallyAvailableAt = release_date
            self.originallyAvailableAt = release_date
            self.guid = "shared-guid"
            self.ratingKey = None

    newer = StubTrack("Album Remaster", 80, "2020-06-01")
    older = StubTrack("Album Original", 10, "1980-2-2")

    log = logging.getLogger("test_deduplicate_oldest_non_padded")
    log.setLevel(logging.DEBUG)

    deduped, _, removed, reason_counts = main._deduplicate_tracks(
        [newer, older],
        log,
        duplicate_tiebreaker="oldest",
    )

    assert removed == 1
    assert [track.parentTitle for track in deduped] == ["Album Original"]
    assert reason_counts == {"oldest": 1}


def test_deduplicate_tracks_prefers_newest_when_configured():
    import main

    class StubTrack:
        def __init__(self, album, popularity, release_date):
            self.title = "Song"
            self.grandparentTitle = "Artist"
            self.parentTitle = album
            self.ratingCount = popularity
            self.parentRatingCount = popularity
            self.parentOriginallyAvailableAt = release_date
            self.originallyAvailableAt = release_date
            self.guid = "shared-guid"
            self.ratingKey = None

    older = StubTrack("Album Original", 80, "1975-01-01")
    newer = StubTrack("Album Remaster", 5, "2022-09-09")

    log = logging.getLogger("test_deduplicate_newest")
    log.setLevel(logging.DEBUG)

    deduped, _, removed, reason_counts = main._deduplicate_tracks(
        [older, newer],
        log,
        duplicate_tiebreaker="newest",
    )

    assert removed == 1
    assert [track.parentTitle for track in deduped] == ["Album Remaster"]
    assert reason_counts == {"newest": 1}


def test_deduplicate_tracks_allow_keeps_duplicates():
    import main

    class StubTrack:
        def __init__(self, title, popularity):
            self.title = title
            self.grandparentTitle = "Artist"
            self.parentTitle = "Album"
            self.ratingCount = popularity
            self.parentRatingCount = popularity
            self.parentOriginallyAvailableAt = "2000-01-01"
            self.originallyAvailableAt = "2000-01-01"
            self.guid = "shared-guid"
            self.ratingKey = None

    first = StubTrack("Song 1", 5)
    second = StubTrack("Song 2", 10)

    log = logging.getLogger("test_deduplicate_allow")
    log.setLevel(logging.DEBUG)

    deduped, cache, removed, reason_counts = main._deduplicate_tracks(
        [first, second],
        log,
        duplicate_tiebreaker="allow",
    )

    assert removed == 0
    assert deduped == [first, second]
    assert cache == {}
    assert reason_counts == {}


def test_resolve_album_release_date_fetches_metadata(monkeypatch):
    import main

    class _Track:
        parentOriginallyAvailableAt = None
        originallyAvailableAt = None
        ratingKey = 111
        parentRatingKey = 222

    track = _Track()

    main._ALBUM_RELEASE_DATE_CACHE.clear()
    main._ALBUM_RELEASE_DATE_MISS_KEYS.clear()

    fetched_keys = []

    def _fake_fetch_full_metadata(key):
        fetched_keys.append(key)
        if key == track.ratingKey:
            return (
                "<MediaContainer><Track parentOriginallyAvailableAt=\"1990-10-17\" />"
                "</MediaContainer>"
            )
        if key == track.parentRatingKey:
            return "<MediaContainer><Directory originallyAvailableAt=\"1990-11-01\" /></MediaContainer>"
        raise AssertionError(f"Unexpected metadata lookup for key={key}")

    monkeypatch.setattr(main, "fetch_full_metadata", _fake_fetch_full_metadata)

    result = main._resolve_album_release_date(track)

    assert result == datetime.date(1990, 10, 17)
    assert fetched_keys == [track.ratingKey]

    fetched_keys.clear()

    # Subsequent lookups should be served from cache without extra fetches.
    result_again = main._resolve_album_release_date(track)

    assert result_again == datetime.date(1990, 10, 17)
    assert fetched_keys == []
