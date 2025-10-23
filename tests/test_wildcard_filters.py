import importlib
import logging

import pytest


class DummyTrack:
    def __init__(self, **fields):
        self._fields = fields


@pytest.fixture(scope="module")
def main_module():
    import plexapi.server as plex_server_module

    patcher = pytest.MonkeyPatch()

    class _FakePlex:
        def library(self, *args, **kwargs):
            return None

    patcher.setattr(plex_server_module, "PlexServer", lambda *args, **kwargs: _FakePlex())
    module = importlib.import_module("main")
    yield module
    patcher.undo()


@pytest.fixture(autouse=True)
def stub_get_field_value(monkeypatch, main_module):
    def _fake_get_field_value(track, field):
        return track._fields.get(field)

    monkeypatch.setattr(main_module, "get_field_value", _fake_get_field_value)


def _evaluate(module, track, wildcard_filters, regular_filters):
    logger = logging.getLogger("test")
    return module._evaluate_track_filters(track, wildcard_filters, regular_filters, logger, False)


def test_wildcard_match_allows_skipping_other_filters(main_module):
    track = DummyTrack(artist="Foxy Shazam", **{"album.year": 1950})
    wildcard_filters = [
        {"field": "artist", "operator": "equals", "value": "Foxy Shazam"}
    ]
    regular_filters = [
        {"field": "album.year", "operator": "greater_than", "value": 1963}
    ]

    keep, wildcard_matched = _evaluate(main_module, track, wildcard_filters, regular_filters)

    assert keep is True
    assert wildcard_matched is True


def test_regular_filters_still_apply_when_wildcard_not_matched(main_module):
    track = DummyTrack(artist="Other Artist", **{"album.year": 1970})
    wildcard_filters = [
        {"field": "artist", "operator": "equals", "value": "Foxy Shazam"}
    ]
    regular_filters = [
        {"field": "album.year", "operator": "greater_than", "value": 1963}
    ]

    keep, wildcard_matched = _evaluate(main_module, track, wildcard_filters, regular_filters)

    assert keep is True
    assert wildcard_matched is False


def test_track_removed_when_no_filters_match(main_module):
    track = DummyTrack(artist="Other Artist", **{"album.year": 1950})
    wildcard_filters = [
        {"field": "artist", "operator": "equals", "value": "Foxy Shazam"}
    ]
    regular_filters = [
        {"field": "album.year", "operator": "greater_than", "value": 1963}
    ]

    keep, wildcard_matched = _evaluate(main_module, track, wildcard_filters, regular_filters)

    assert keep is False
    assert wildcard_matched is False


def test_only_wildcard_filters_must_match(main_module):
    track = DummyTrack(artist="Other Artist")
    wildcard_filters = [
        {"field": "artist", "operator": "equals", "value": "Foxy Shazam"}
    ]

    keep, wildcard_matched = _evaluate(main_module, track, wildcard_filters, [])

    assert keep is False
    assert wildcard_matched is False


def test_no_filters_keeps_track(main_module):
    track = DummyTrack(artist="Anyone")

    keep, wildcard_matched = _evaluate(main_module, track, [], [])

    assert keep is True
    assert wildcard_matched is False


def test_negative_wildcard_filters_block_when_any_expected_matches(main_module):
    track = DummyTrack(genres=["Album Rock", "Arena Rock"])
    wildcard_filters = [
        {
            "field": "genres",
            "operator": "does_not_equal",
            "value": ["AOR", "Arena Rock"],
            "match_all": False,
            "wildcard": True,
        }
    ]

    keep, wildcard_matched = _evaluate(main_module, track, wildcard_filters, [])

    assert keep is False
    assert wildcard_matched is False


def test_negative_regular_filters_block_when_any_expected_matches(main_module):
    track = DummyTrack(title="My Song (Explicit Version)")
    regular_filters = [
        {
            "field": "title",
            "operator": "does_not_contain",
            "value": ["explicit", "clean"],
            "match_all": False,
        }
    ]

    keep, wildcard_matched = _evaluate(main_module, track, [], regular_filters)

    assert keep is False
    assert wildcard_matched is False
