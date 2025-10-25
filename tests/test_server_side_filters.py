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
    assert stats["requests"] == 10
    assert stats["original_count"] == 7
    assert stats["duplicates_removed"] == 2
    assert library.calls == [
        {"libtype": "track", "filters": {"track.genre": "Rock"}},
        {"libtype": "track", "filters": {"album.genre": "Rock"}},
        {"libtype": "track", "filters": {"album.style": "Rock"}},
        {"libtype": "track", "filters": {"artist.genre": "Rock"}},
        {"libtype": "track", "filters": {"artist.style": "Rock"}},
        {"libtype": "track", "filters": {"track.genre": "Metal"}},
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
    assert stats["requests"] == 20
    assert stats["duplicates_removed"] == 0
    assert library.calls == [
        {"libtype": "track", "filters": {"track.genre": "Rock", "track.mood": "Happy"}},
        {"libtype": "track", "filters": {"album.genre": "Rock", "album.mood": "Happy"}},
        {"libtype": "track", "filters": {"album.style": "Rock", "album.mood": "Happy"}},
        {"libtype": "track", "filters": {"artist.genre": "Rock", "artist.mood": "Happy"}},
        {"libtype": "track", "filters": {"artist.style": "Rock", "artist.mood": "Happy"}},
        {"libtype": "track", "filters": {"track.genre": "Rock", "track.mood": "Moody"}},
        {"libtype": "track", "filters": {"album.genre": "Rock", "album.mood": "Moody"}},
        {"libtype": "track", "filters": {"album.style": "Rock", "album.mood": "Moody"}},
        {"libtype": "track", "filters": {"artist.genre": "Rock", "artist.mood": "Moody"}},
        {"libtype": "track", "filters": {"artist.style": "Rock", "artist.mood": "Moody"}},
        {"libtype": "track", "filters": {"track.genre": "Metal", "track.mood": "Happy"}},
        {"libtype": "track", "filters": {"album.genre": "Metal", "album.mood": "Happy"}},
        {"libtype": "track", "filters": {"album.style": "Metal", "album.mood": "Happy"}},
        {"libtype": "track", "filters": {"artist.genre": "Metal", "artist.mood": "Happy"}},
        {"libtype": "track", "filters": {"artist.style": "Metal", "artist.mood": "Happy"}},
        {"libtype": "track", "filters": {"track.genre": "Metal", "track.mood": "Moody"}},
        {"libtype": "track", "filters": {"album.genre": "Metal", "album.mood": "Moody"}},
        {"libtype": "track", "filters": {"album.style": "Metal", "album.mood": "Moody"}},
        {"libtype": "track", "filters": {"artist.genre": "Metal", "artist.mood": "Moody"}},
        {"libtype": "track", "filters": {"artist.style": "Metal", "artist.mood": "Moody"}},
    ]


def test_fetch_tracks_includes_track_level_queries_for_styles():
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

    assert [track.ratingKey for track in tracks] == ["track-1", "album-1", "artist-1"]
    assert stats["requests"] == 3
    assert library.calls == [
        {"libtype": "track", "filters": {"track.style": "Shoegaze"}},
        {"libtype": "track", "filters": {"album.style": "Shoegaze"}},
        {"libtype": "track", "filters": {"artist.style": "Shoegaze"}},
    ]

