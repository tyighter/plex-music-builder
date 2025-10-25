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
        for field in ("album.genre", "artist.genre", "track.genre"):
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
    }
    library = _RecordingLibrary(responses)

    multi_filters = [
        {"key_type": "filters", "key_name": "track.genre", "values": ["Rock", "Metal"]}
    ]

    tracks, stats = main._fetch_tracks_with_server_filters(library, {}, {}, multi_filters)

    assert [track.ratingKey for track in tracks] == ["1", "2", "3"]
    assert stats["requests"] == 4
    assert stats["original_count"] == 4
    assert stats["duplicates_removed"] == 1
    assert [
        (
            next(
                field for field in ("album.genre", "artist.genre") if field in call.get("filters", {})
            ),
            call["filters"][
                next(
                    field for field in ("album.genre", "artist.genre") if field in call.get("filters", {})
                )
            ],
        )
        for call in library.calls
    ] == [
        ("album.genre", "Rock"),
        ("artist.genre", "Rock"),
        ("album.genre", "Metal"),
        ("artist.genre", "Metal"),
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
    assert stats["requests"] == 8
    assert stats["duplicates_removed"] == 0
    assert {
        (
            call.get("filters", {}).get("album.genre"),
            call.get("filters", {}).get("album.mood"),
            call.get("filters", {}).get("artist.genre"),
            call.get("filters", {}).get("artist.mood"),
        )
        for call in library.calls
    } == {
        ("Rock", "Happy", None, None),
        ("Rock", "Moody", None, None),
        ("Metal", "Happy", None, None),
        ("Metal", "Moody", None, None),
        (None, None, "Rock", "Happy"),
        (None, None, "Rock", "Moody"),
        (None, None, "Metal", "Happy"),
        (None, None, "Metal", "Moody"),
    }

