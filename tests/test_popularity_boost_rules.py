import sys
import types
from types import SimpleNamespace

import pytest

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

from main import _apply_configured_popularity_boosts, _resolve_popularity_for_sort


class DummyTrack(SimpleNamespace):
    def __init__(self, rating_key="1", genre="Rock", artist="Example Artist"):
        super().__init__()
        self.ratingKey = rating_key
        self.genre = genre
        self.grandparentTitle = artist
        self.title = "Example Track"


def test_popularity_boost_updates_dedup_cache():
    track = DummyTrack(rating_key="1", genre="Rock")
    dedup_popularity_cache = {"1": 50.0}
    boosts = [
        {"field": "genre", "operator": "equals", "value": "Rock", "boost": 2.0}
    ]

    _apply_configured_popularity_boosts(
        [track],
        boosts,
        dedup_popularity_cache,
        {},
        {},
        playlist_logger=None,
    )

    assert dedup_popularity_cache["1"] == pytest.approx(100.0)


def test_comma_separated_values_use_match_any():
    track = DummyTrack(rating_key="5", genre="Rock")
    dedup_popularity_cache = {"5": 10.0}
    boosts = [
        {
            "field": "genre",
            "operator": "equals",
            "value": "Rock, Pop",
            "boost": 3.0,
        }
    ]

    _apply_configured_popularity_boosts(
        [track],
        boosts,
        dedup_popularity_cache,
        {},
        {},
        playlist_logger=None,
    )

    assert dedup_popularity_cache["5"] == pytest.approx(30.0)


def test_comma_separated_values_override_explicit_match_all_true():
    track = DummyTrack(rating_key="6", genre="Rock")
    dedup_popularity_cache = {"6": 12.0}
    boosts = [
        {
            "field": "genre",
            "operator": "equals",
            "value": "Rock, Pop",
            "boost": 2.0,
            "match_all": True,
        }
    ]

    _apply_configured_popularity_boosts(
        [track],
        boosts,
        dedup_popularity_cache,
        {},
        {},
        playlist_logger=None,
    )

    assert dedup_popularity_cache["6"] == pytest.approx(24.0)


def test_multiple_rules_stack_multiplier_and_update_album_cache():
    track = DummyTrack(rating_key="2", artist="The Rockets")
    dedup_popularity_cache = {"2": 10.0}
    album_popularity_cache = {"2": 10.0}
    boosts = [
        {"field": "genre", "operator": "equals", "value": "Rock", "boost": 2.0},
        {
            "field": "grandparentTitle",
            "operator": "contains",
            "value": "rocket",
            "boost": 1.5,
        },
    ]

    _apply_configured_popularity_boosts(
        [track],
        boosts,
        dedup_popularity_cache,
        album_popularity_cache,
        {},
        playlist_logger=None,
    )

    assert dedup_popularity_cache["2"] == pytest.approx(30.0)
    assert album_popularity_cache["2"] == pytest.approx(30.0)


def test_invalid_multiplier_defaults_to_one():
    track = DummyTrack(rating_key="3")
    dedup_popularity_cache = {"3": 25.0}
    boosts = [
        {"field": "genre", "operator": "equals", "value": "Rock", "boost": "invalid"}
    ]

    _apply_configured_popularity_boosts(
        [track],
        boosts,
        dedup_popularity_cache,
        {},
        {},
        playlist_logger=None,
    )

    assert dedup_popularity_cache["3"] == pytest.approx(25.0)


def test_boost_applies_to_tracks_without_rating_key():
    track = DummyTrack(rating_key=None)
    album_popularity_cache_by_object = {id(track): 15.0}
    boosts = [
        {"field": "genre", "operator": "equals", "value": "Rock", "boost": 2.0}
    ]

    _apply_configured_popularity_boosts(
        [track],
        boosts,
        {},
        {},
        album_popularity_cache_by_object,
        playlist_logger=None,
    )

    assert album_popularity_cache_by_object[id(track)] == pytest.approx(30.0)


def test_sorting_prefers_album_boost_over_dedup_cache():
    track = DummyTrack(rating_key="4")
    dedup_popularity_cache = {"4": 40.0}
    album_popularity_cache = {"4": 80.0}

    value, has_value = _resolve_popularity_for_sort(
        track,
        "4",
        id(track),
        dedup_popularity_cache,
        album_popularity_cache,
        {},
        playlist_logger=None,
        sort_desc=True,
    )

    assert has_value is True
    assert value == pytest.approx(80.0)
    assert dedup_popularity_cache["4"] == pytest.approx(80.0)
