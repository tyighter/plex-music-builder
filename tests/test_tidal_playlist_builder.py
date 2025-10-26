import pytest

import main


class _DummyTrack:
    def __init__(self, title, album, artist, rating_count, rating_key):
        self.title = title
        self.parentTitle = album
        self.grandparentTitle = artist
        self.ratingCount = rating_count
        self.ratingKey = rating_key


class _DummyLibrary:
    def __init__(self, responses):
        self._responses = {
            self._normalize_kwargs(kwargs): list(results)
            for kwargs, results in responses
        }
        self.calls = []

    @staticmethod
    def _normalize_kwargs(kwargs):
        items = []
        for key, value in sorted(kwargs.items()):
            if isinstance(value, dict):
                items.append((key, tuple(sorted(value.items()))))
            else:
                items.append((key, value))
        return tuple(items)

    def search(self, **kwargs):
        normalized = self._normalize_kwargs(kwargs)
        self.calls.append(("search", kwargs))
        return list(self._responses.get(normalized, []))


class _DummyLog:
    def __init__(self):
        self.records = []

    def info(self, *args, **kwargs):
        self.records.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.records.append(("warning", args, kwargs))

    def debug(self, *args, **kwargs):  # pragma: no cover - optional tracing
        pass

    def isEnabledFor(self, level):
        return False


def test_normalize_tidal_playlist_url_variants():
    playlist_id = "B9D516BC-0BAF-40B0-9152-37D2D9B86C55"
    assert (
        main._normalize_tidal_playlist_url(
            f"https://tidal.com/browse/playlist/{playlist_id}?foo=bar"
        )
        == "https://tidal.com/browse/playlist/b9d516bc-0baf-40b0-9152-37d2d9b86c55"
    )
    assert (
        main._normalize_tidal_playlist_url(playlist_id)
        == "https://tidal.com/browse/playlist/b9d516bc-0baf-40b0-9152-37d2d9b86c55"
    )
    with pytest.raises(ValueError):
        main._normalize_tidal_playlist_url("https://example.com/not-tidal")


def test_collect_tidal_tracks_handles_pagination(monkeypatch):
    playlist_id = "b9d516bc-0baf-40b0-9152-37d2d9b86c55"

    track_one = _DummyTrack("Song One", "Album One", "Artist One", 10, 1)
    track_two = _DummyTrack("Song Two", "Album Two", "Artist Two", 5, 2)

    responses = [
        (
            {"libtype": "track", "filters": {"artist.title": "Artist One"}},
            [track_one],
        ),
        (
            {"libtype": "track", "filters": {"artist.title": "Artist Two"}},
            [track_two],
        ),
    ]
    library = _DummyLibrary(responses)
    log = _DummyLog()

    page_payloads = [
        {
            "items": [
                {
                    "item": {
                        "title": "Song One",
                        "album": {"title": "Album One"},
                        "artists": [{"name": "Artist One"}],
                    }
                }
            ],
            "totalNumberOfItems": 2,
            "limit": 1,
        },
        {
            "items": [
                {
                    "item": {
                        "title": "Song Two",
                        "album": {"title": "Album Two"},
                        "artists": [{"name": "Artist Two"}],
                    }
                }
            ],
            "totalNumberOfItems": 2,
            "limit": 1,
        },
    ]

    fetched_offsets = []

    def fake_fetch(page_playlist_id, offset, logger):
        assert page_playlist_id == playlist_id
        fetched_offsets.append(offset)
        if not page_payloads:
            raise AssertionError("Unexpected additional Tidal page fetch")
        return page_payloads.pop(0)

    monkeypatch.setattr(main, "_fetch_tidal_tracks_page", fake_fetch)

    matched, stats = main._collect_tidal_tracks(
        f"https://tidal.com/browse/playlist/{playlist_id}",
        library,
        log,
    )

    assert matched == [track_one, track_two]
    assert stats == {
        "normalized_url": f"https://tidal.com/browse/playlist/{playlist_id}",
        "total_tracks": 2,
        "matched_tracks": 2,
        "unmatched_tracks": 0,
    }
    assert fetched_offsets == [0, 1]
    assert library.calls == [
        ("search", {"libtype": "track", "filters": {"artist.title": "Artist One"}}),
        ("search", {"libtype": "track", "filters": {"artist.title": "Artist Two"}}),
    ]
