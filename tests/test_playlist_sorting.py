from collections import OrderedDict

import yaml


def write_yaml(path, data):
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def test_save_playlists_alphabetizes_entries(tmp_path, monkeypatch):
    from gui import save_playlists

    playlist_path = tmp_path / "playlists.yml"
    monkeypatch.setattr("gui.PLAYLISTS_PATH", playlist_path)

    payload = {
        "defaults": {},
        "playlists": [
            {"name": "Rock", "limit": 0, "artist_limit": 0, "album_limit": 0},
            {"name": "acoustic", "limit": 0, "artist_limit": 0, "album_limit": 0},
            {"name": "Blues", "limit": 0, "artist_limit": 0, "album_limit": 0},
        ],
    }

    save_playlists(payload)

    with playlist_path.open("r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert list(saved["playlists"].keys()) == ["acoustic", "Blues", "Rock"]


def test_save_single_playlist_preserves_sorted_order(tmp_path, monkeypatch):
    from gui import save_single_playlist

    playlist_path = tmp_path / "playlists.yml"
    monkeypatch.setattr("gui.PLAYLISTS_PATH", playlist_path)

    initial_data = {
        "defaults": {},
        "playlists": OrderedDict(
            [
                ("Rock", {"limit": 0, "artist_limit": 0, "album_limit": 0}),
                ("acoustic", {"limit": 0, "artist_limit": 0, "album_limit": 0}),
            ]
        ),
    }
    write_yaml(playlist_path, initial_data)

    save_single_playlist({"name": "Blues", "limit": 0, "artist_limit": 0, "album_limit": 0})

    with playlist_path.open("r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert list(saved["playlists"].keys()) == ["acoustic", "Blues", "Rock"]
