from types import SimpleNamespace

import pytest

from main import _apply_configured_popularity_boosts


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
