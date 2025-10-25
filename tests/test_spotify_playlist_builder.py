import json
from urllib.parse import quote

import pytest
import requests

import main


def test_normalize_spotify_playlist_url_variants():
    assert (
        main._normalize_spotify_playlist_url(
            "https://open.spotify.com/playlist/abc123?si=xyz"
        )
        == "https://open.spotify.com/playlist/abc123"
    )
    assert (
        main._normalize_spotify_playlist_url("spotify:playlist:AbC456")
        == "https://open.spotify.com/playlist/AbC456"
    )
    with pytest.raises(ValueError):
        main._normalize_spotify_playlist_url("https://example.com/not-spotify")


def test_parse_spotify_entity_tracks_skips_local_and_missing():
    entity = {
        "tracks": {
            "items": [
                {"track": {"name": "Local Song", "is_local": True}},
                {
                    "track": {
                        "name": "Cloud Song",
                        "album": {"name": "Great Album", "artists": [{"name": "Artist"}]},
                    }
                },
                {"track": None},
            ]
        }
    }

    parsed = main._parse_spotify_entity_tracks(entity)
    assert parsed == [{"title": "Cloud Song", "artist": "Artist"}]


def test_extract_spotify_entity_payload_handles_json_parse_decode():
    payload = {"tracks": {"items": []}}
    encoded = quote(json.dumps(payload))
    html = f'<script>Spotify.Entity = JSON.parse(decodeURIComponent("{encoded}"));</script>'

    parsed = main._extract_spotify_entity_payload(html)

    assert parsed == payload


def test_extract_spotify_entity_payload_handles_json_parse_string():
    payload = {"tracks": {"items": []}}
    html_literal = json.dumps(json.dumps(payload))
    html = f"<script>Spotify.Entity = JSON.parse({html_literal});</script>"

    parsed = main._extract_spotify_entity_payload(html)

    assert parsed == payload


def test_extract_spotify_entity_payload_handles_next_data_tracks():
    next_data = {
        "props": {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "state": {
                                "data": {
                                    "playlistV2": {
                                        "trackList": {
                                            "items": [
                                                {
                                                    "itemV2": {
                                                        "data": {
                                                            "__typename": "Track",
                                                            "uri": "spotify:track:123",
                                                            "name": "Next Song",
                                                            "albumOfTrack": {
                                                                "name": "Next Album",
                                                                "artists": {
                                                                    "items": [
                                                                        {
                                                                            "profile": {
                                                                                "name": "Next Artist"
                                                                            }
                                                                        }
                                                                    ]
                                                                },
                                                            },
                                                        }
                                                    }
                                                },
                                                {
                                                    "itemV2": {
                                                        "data": {
                                                            "__typename": "Track",
                                                            "uri": "spotify:track:456",
                                                            "name": "Another Song",
                                                            "albumOfTrack": {"name": "Another Album"},
                                                            "artists": {
                                                                "items": [
                                                                    {
                                                                        "profile": {
                                                                            "name": "Another Artist"
                                                                        }
                                                                    }
                                                                ]
                                                            },
                                                        }
                                                    }
                                                },
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }
    }

    html = (
        '<html><body><script id="__NEXT_DATA__" type="application/json">'
        f"{json.dumps(next_data)}</script></body></html>"
    )

    parsed = main._extract_spotify_entity_payload(html)

    assert parsed == {
        "tracks": {
            "items": [
                {
                    "track": {
                        "name": "Next Song",
                        "album": {
                            "name": "Next Album",
                            "artists": [{"name": "Next Artist"}],
                        },
                        "artists": [{"name": "Next Artist"}],
                        "uri": "spotify:track:123",
                    }
                },
                {
                    "track": {
                        "name": "Another Song",
                        "album": {
                            "name": "Another Album",
                            "artists": [{"name": "Another Artist"}],
                        },
                        "artists": [{"name": "Another Artist"}],
                        "uri": "spotify:track:456",
                    }
                },
            ]
        }
    }

    tracks = main._parse_spotify_entity_tracks(parsed)

    assert tracks == [
        {"title": "Next Song", "artist": "Next Artist"},
        {"title": "Another Song", "artist": "Another Artist"},
    ]


class _DummyTrack:
    def __init__(self, title, album, artist, rating_count, rating_key):
        self.title = title
        self.parentTitle = album
        self.grandparentTitle = artist
        self.ratingCount = rating_count
        self.ratingKey = rating_key


class _DummyLibrary:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    def searchTracks(self, **kwargs):
        self.calls.append(kwargs)
        key = tuple(sorted(kwargs.items()))
        return list(self._responses.get(key, []))


class _DummyLog:
    def __init__(self):
        self.records = []

    def debug(self, *args, **kwargs):  # pragma: no cover - optional tracing
        pass

    def info(self, *args, **kwargs):
        self.records.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.records.append(("warning", args, kwargs))

    def isEnabledFor(self, level):
        return False


class _FakeResponse:
    def __init__(self, text, status_code=200, headers=None):
        self._text = text
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html"}

    @property
    def text(self):
        return self._text

    @property
    def content(self):
        return self._text.encode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"Status {self.status_code}", response=self)


def test_match_spotify_tracks_prefers_higher_ratingcount():
    track_low = _DummyTrack("Song", "Album", "Artist", 5, 1)
    track_high = _DummyTrack("Song", "Album", "Artist", 12, 2)

    query = tuple(sorted([("artist", "Artist"), ("title", "Song")]))
    responses = {query: [track_low, track_high]}
    library = _DummyLibrary(responses)
    log = _DummyLog()

    spotify_tracks = [
        {"title": "Song (Remastered)", "artist": "Artist"}
    ]

    matched, unmatched = main._match_spotify_tracks_to_library(
        spotify_tracks,
        library,
        log,
    )

    assert matched == [track_high]
    assert unmatched == 0
    assert library.calls[0] == {"artist": "Artist"}
    assert {"title": "Song", "artist": "Artist"} in library.calls


def test_collect_spotify_tracks_falls_back_to_embed(monkeypatch):
    login_page = "<html><head><title>Spotify – Web Player</title></head><body></body></html>"
    entity_payload = {
        "tracks": {
            "items": [
                {
                    "track": {
                        "name": "Song",
                        "album": {
                            "name": "Album",
                            "artists": [{"name": "Artist"}],
                        },
                        "artists": [{"name": "Artist"}],
                    }
                }
            ]
        }
    }
    embed_page = f"<script>Spotify.Entity = {json.dumps(entity_payload)};</script>"

    responses = [
        _FakeResponse(login_page),
        _FakeResponse(embed_page),
    ]
    requested_urls = []

    def fake_get(url, headers=None, timeout=None):
        requested_urls.append(url)
        if not responses:
            raise AssertionError("Unexpected additional Spotify requests")
        return responses.pop(0)

    monkeypatch.setattr(main, "requests", requests)
    monkeypatch.setattr(main.requests, "get", fake_get)

    track = _DummyTrack("Song", "Album", "Artist", 10, 1)
    query = tuple(sorted([("artist", "Artist"), ("title", "Song")]))
    library = _DummyLibrary({query: [track]})
    log = _DummyLog()

    matched, stats = main._collect_spotify_tracks(
        "https://open.spotify.com/playlist/37i9dQZF1EQpVaHRDcozEz",
        library,
        log,
    )

    assert matched == [track]
    assert stats == {
        "normalized_url": "https://open.spotify.com/playlist/37i9dQZF1EQpVaHRDcozEz",
        "total_tracks": 1,
        "matched_tracks": 1,
        "unmatched_tracks": 0,
    }
    assert requested_urls == [
        "https://open.spotify.com/playlist/37i9dQZF1EQpVaHRDcozEz",
        "https://open.spotify.com/embed/playlist/37i9dQZF1EQpVaHRDcozEz",
    ]

