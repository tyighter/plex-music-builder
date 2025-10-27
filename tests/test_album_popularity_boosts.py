from __future__ import annotations

import sys
import types
from typing import List, Tuple

if "plexapi.server" not in sys.modules:
    plexapi_module = types.ModuleType("plexapi")
    plexapi_server_module = types.ModuleType("plexapi.server")

    class _DummyPlexServer:  # pragma: no cover - trivial stub
        def __init__(self, *args, **kwargs) -> None:
            pass

    plexapi_server_module.PlexServer = _DummyPlexServer
    plexapi_module.server = plexapi_server_module
    sys.modules["plexapi"] = plexapi_module
    sys.modules["plexapi.server"] = plexapi_server_module

from main import _compute_album_popularity_boosts


class DummyTrack:
    def __init__(self, rating_key: int, album_key: str = "album-1") -> None:
        self.ratingKey = rating_key
        self.parentRatingKey = album_key
        self.parentGuid = None
        self.parentKey = None
        self.grandparentTitle = "Artist"
        self.originalTitle = None
        self.parentTitle = "Album"
        self.parentYear = None
        self.year = None
        self.originallyAvailableAt = None


def _build_tracks_and_cache() -> Tuple[List[DummyTrack], dict]:
    base_scores = [100, 90, 80, 70, 60, 50]
    tracks = [DummyTrack(index) for index in range(1, len(base_scores) + 1)]
    popularity_cache = {
        str(index): score for index, score in enumerate(base_scores, start=1)
    }
    return tracks, popularity_cache


def test_album_popularity_cache_uses_existing_scores():
    tracks, popularity_cache = _build_tracks_and_cache()

    adjusted_by_rating_key, adjusted_by_object = _compute_album_popularity_boosts(
        tracks,
        popularity_cache,
    )

    assert adjusted_by_object == {}
    expected = {
        "1": 100.0,
        "2": 90.0,
        "3": 80.0,
        "4": 70.0,
        "5": 60.0,
        "6": 50,
    }
    assert adjusted_by_rating_key == expected


def test_album_popularity_cache_returns_empty_structures_for_no_tracks():
    adjusted_by_rating_key, adjusted_by_object = _compute_album_popularity_boosts(
        [],
        {},
    )

    assert adjusted_by_rating_key == {}
    assert adjusted_by_object == {}
