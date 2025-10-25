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

    server_kwargs, server_filters = main._build_server_side_search_filters([compiled])

    assert server_kwargs == {}
    assert server_filters == {"track.genre": ["Rock", "Metal"]}


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

    server_kwargs, server_filters = main._build_server_side_search_filters([compiled])

    assert server_kwargs == {}
    assert server_filters == {"track.genre&": ["Rock", "Metal"]}


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

    server_kwargs, server_filters = main._build_server_side_search_filters([compiled])

    assert server_kwargs == {}
    assert server_filters == {"track.genre": ["Rock", "Metal"]}
