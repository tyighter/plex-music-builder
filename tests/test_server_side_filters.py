import logging
import importlib


def _load_main_module():
    return importlib.import_module("main")


def test_server_filters_respect_match_all_false_string():
    main = _load_main_module()

    compiled = main._compile_filter_entry(
        {
            "field": "genre",
            "operator": "equals",
            "value": ["Rock", "Metal"],
            "match_all": "false",
        }
    )

    server_kwargs, server_filters, multi_filters = main._build_server_side_search_filters([compiled])

    assert server_kwargs == {}
    assert server_filters == {}
    assert multi_filters == [
        {"key_type": "filters", "key_name": "track.genre", "values": ["Rock", "Metal"]}
    ]


def test_server_filters_respect_match_all_true_string():
    main = _load_main_module()

    compiled = main._compile_filter_entry(
        {
            "field": "genre",
            "operator": "equals",
            "value": ["Rock", "Metal"],
            "match_all": "TRUE",
        }
    )

    server_kwargs, server_filters, multi_filters = main._build_server_side_search_filters([compiled])

    assert server_kwargs == {}
    assert server_filters == {"track.genre&": ["Rock", "Metal"]}
    assert multi_filters == []


def test_server_filters_default_to_any_match_for_multiple_values():
    main = _load_main_module()

    compiled = main._compile_filter_entry(
        {
            "field": "genre",
            "operator": "equals",
            "value": ["Rock", "Metal"],
        }
    )

    assert compiled.match_all is False

    server_kwargs, server_filters, multi_filters = main._build_server_side_search_filters([compiled])

    assert server_kwargs == {}
    assert server_filters == {}
    assert multi_filters == [
        {"key_type": "filters", "key_name": "track.genre", "values": ["Rock", "Metal"]}
    ]


def test_server_filters_map_artist_title_to_server_query():
    main = _load_main_module()

    compiled = main._compile_filter_entry(
        {
            "field": "artist",
            "operator": "equals",
            "value": "The Beatles",
        }
    )

    server_kwargs, server_filters, multi_filters = main._build_server_side_search_filters([compiled])

    assert server_kwargs == {}
    assert server_filters == {"artist.title": "The Beatles"}
    assert multi_filters == []


class _DummyTrack:
    def __init__(self, rating_key: str):
        self.ratingKey = rating_key
        self.guid = f"guid-{rating_key}"


class _RecordingLibrary:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        filters = kwargs.get("filters", {})
        for field in (
            "album.genre",
            "artist.genre",
            "track.genre",
            "album.style",
            "artist.style",
            "track.style",
        ):
            if field in filters:
                lookup_key = (field, filters[field])
                return list(self._responses.get(lookup_key, []))
        return []


class _CombinationLibrary:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        filters = kwargs.get("filters", {})
        genre_field = None
        mood_field = None
        for field in ("album.genre", "artist.genre", "track.genre"):
            if field in filters:
                genre_field = field
                genre = filters[field]
                break
        for field in ("album.mood", "artist.mood", "track.mood"):
            if field in filters:
                mood_field = field
                mood = filters[field]
                break

        if genre_field == "album.genre" and mood_field == "album.mood":
            key = f"{genre}-{mood}"
            return [_DummyTrack(key)]
        return []


class _UnionLibrary:
    def __init__(self, beatles_key: str = "beatles-1", wildcard_key: str = "wildcard-1"):
        self.calls = []
        self.search_tracks_called = False
        self._beatles_key = beatles_key
        self._wildcard_key = wildcard_key

    def search(self, **kwargs):
        self.calls.append(kwargs)
        filters = kwargs.get("filters", {})
        if filters.get("artist.title") == "The Beatles":
            return [_DummyTrack(self._beatles_key)]
        if filters.get("album.genre") == "Jazz":
            return [_DummyTrack(self._wildcard_key)]
        return []

    def searchTracks(self):
        self.search_tracks_called = True
        return []


def test_fetch_tracks_expands_multi_value_filters_and_deduplicates():
    main = _load_main_module()

    responses = {
        ("album.genre", "Rock"): [_DummyTrack("1"), _DummyTrack("2")],
        ("album.genre", "Metal"): [_DummyTrack("2"), _DummyTrack("3")],
        ("album.style", "Rock"): [_DummyTrack("1"), _DummyTrack("4")],
        ("artist.genre", "Metal"): [_DummyTrack("5")],
    }
    library = _RecordingLibrary(responses)

    multi_filters = [
        {"key_type": "filters", "key_name": "track.genre", "values": ["Rock", "Metal"]}
    ]

    tracks, stats = main._fetch_tracks_with_server_filters(library, {}, {}, multi_filters)

    assert [track.ratingKey for track in tracks] == ["1", "2", "4", "3", "5"]
    assert stats["requests"] == 8
    assert stats["original_count"] == 7
    assert stats["duplicates_removed"] == 2
    assert library.calls == [
        {"libtype": "track", "filters": {"album.genre": "Rock"}},
        {"libtype": "track", "filters": {"album.style": "Rock"}},
        {"libtype": "track", "filters": {"artist.genre": "Rock"}},
        {"libtype": "track", "filters": {"artist.style": "Rock"}},
        {"libtype": "track", "filters": {"album.genre": "Metal"}},
        {"libtype": "track", "filters": {"album.style": "Metal"}},
        {"libtype": "track", "filters": {"artist.genre": "Metal"}},
        {"libtype": "track", "filters": {"artist.style": "Metal"}},
    ]


def test_fetch_tracks_generates_combinations_for_multiple_multi_filters():
    main = _load_main_module()

    library = _CombinationLibrary()
    multi_filters = [
        {"key_type": "filters", "key_name": "track.genre", "values": ["Rock", "Metal"]},
        {"key_type": "filters", "key_name": "track.mood", "values": ["Happy", "Moody"]},
    ]

    tracks, stats = main._fetch_tracks_with_server_filters(library, {}, {}, multi_filters)

    expected_order = [
        "Rock-Happy",
        "Rock-Moody",
        "Metal-Happy",
        "Metal-Moody",
    ]

    assert [track.ratingKey for track in tracks] == expected_order
    assert stats["requests"] == 16
    assert stats["duplicates_removed"] == 0
    assert library.calls == [
        {"libtype": "track", "filters": {"album.genre": "Rock", "album.mood": "Happy"}},
        {"libtype": "track", "filters": {"album.style": "Rock", "album.mood": "Happy"}},
        {"libtype": "track", "filters": {"artist.genre": "Rock", "artist.mood": "Happy"}},
        {"libtype": "track", "filters": {"artist.style": "Rock", "artist.mood": "Happy"}},
        {"libtype": "track", "filters": {"album.genre": "Rock", "album.mood": "Moody"}},
        {"libtype": "track", "filters": {"album.style": "Rock", "album.mood": "Moody"}},
        {"libtype": "track", "filters": {"artist.genre": "Rock", "artist.mood": "Moody"}},
        {"libtype": "track", "filters": {"artist.style": "Rock", "artist.mood": "Moody"}},
        {"libtype": "track", "filters": {"album.genre": "Metal", "album.mood": "Happy"}},
        {"libtype": "track", "filters": {"album.style": "Metal", "album.mood": "Happy"}},
        {"libtype": "track", "filters": {"artist.genre": "Metal", "artist.mood": "Happy"}},
        {"libtype": "track", "filters": {"artist.style": "Metal", "artist.mood": "Happy"}},
        {"libtype": "track", "filters": {"album.genre": "Metal", "album.mood": "Moody"}},
        {"libtype": "track", "filters": {"album.style": "Metal", "album.mood": "Moody"}},
        {"libtype": "track", "filters": {"artist.genre": "Metal", "artist.mood": "Moody"}},
        {"libtype": "track", "filters": {"artist.style": "Metal", "artist.mood": "Moody"}},
    ]


def test_fetch_tracks_replaces_track_level_queries_for_styles():
    main = _load_main_module()

    library = _RecordingLibrary(
        {
            ("track.style", "Shoegaze"): [_DummyTrack("track-1")],
            ("album.style", "Shoegaze"): [_DummyTrack("album-1")],
            ("artist.style", "Shoegaze"): [_DummyTrack("artist-1")],
        }
    )

    server_kwargs = {}
    server_filters = {"track.style": "Shoegaze"}
    multi_filters = []

    tracks, stats = main._fetch_tracks_with_server_filters(
        library,
        server_kwargs,
        server_filters,
        multi_filters,
    )

    assert [track.ratingKey for track in tracks] == ["album-1", "artist-1"]
    assert stats["requests"] == 2
    assert library.calls == [
        {"libtype": "track", "filters": {"album.style": "Shoegaze"}},
        {"libtype": "track", "filters": {"artist.style": "Shoegaze"}},
    ]


def test_prefetch_tracks_fetches_union_of_regular_and_wildcard_filters():
    main = _load_main_module()

    library = _UnionLibrary()

    regular_filters = [
        main._compile_filter_entry(
            {"field": "artist", "operator": "equals", "value": "The Beatles"}
        )
    ]
    wildcard_filters = [
        main._compile_filter_entry(
            {"field": "genre", "operator": "equals", "value": "Jazz", "wildcard": True}
        )
    ]

    tracks, stats = main._prefetch_tracks_for_filters(library, regular_filters, wildcard_filters, logging.getLogger("test"))

    assert {track.ratingKey for track in tracks} == {"beatles-1", "wildcard-1"}
    assert stats is not None and stats["requests"] == 5
    assert stats["duplicates_removed"] == 0
    assert library.search_tracks_called is False


def test_prefetch_tracks_deduplicates_overlapping_results():
    main = _load_main_module()

    library = _UnionLibrary(beatles_key="shared", wildcard_key="shared")

    regular_filters = [
        main._compile_filter_entry(
            {"field": "artist", "operator": "equals", "value": "The Beatles"}
        )
    ]
    wildcard_filters = [
        main._compile_filter_entry(
            {"field": "genre", "operator": "equals", "value": "Jazz", "wildcard": True}
        )
    ]

    tracks, stats = main._prefetch_tracks_for_filters(library, regular_filters, wildcard_filters, logging.getLogger("test"))

    assert [track.ratingKey for track in tracks] == ["shared"]
    assert stats is not None and stats["requests"] == 5
    assert stats["duplicates_removed"] == 1
    assert library.search_tracks_called is False

