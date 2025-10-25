import logging
from collections import OrderedDict

import yaml


def write_yaml(path, data):
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


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
