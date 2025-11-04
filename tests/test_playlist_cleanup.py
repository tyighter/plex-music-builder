import json


def _prepare_state(tmp_path, names):
    state_path = tmp_path / "managed.json"
    state_path.write_text(
        json.dumps({"playlists": list(names)}),
        encoding="utf-8",
    )
    return state_path


def _install_common_patches(monkeypatch, main_module, state_path, playlists):
    monkeypatch.setattr(main_module, "MANAGED_PLAYLISTS_FILE", str(state_path))
    monkeypatch.setattr(main_module, "MAX_WORKERS", 1)
    monkeypatch.setattr(main_module, "playlists_data", dict(playlists))

    processed = []

    def _fake_process(name, config):
        processed.append(name)

    monkeypatch.setattr(main_module, "process_playlist", _fake_process)

    return processed


def test_run_playlists_deletes_stale_playlist_when_renamed(tmp_path, monkeypatch):
    import main

    state_path = _prepare_state(tmp_path, ["Old Name"])
    processed = _install_common_patches(monkeypatch, main, state_path, {"New Name": {}})

    deleted = []

    class _DummyPlaylist:
        def __init__(self, title):
            self.title = title

        def delete(self):
            deleted.append(self.title)

    class _DummyServer:
        def playlist(self, name):
            return _DummyPlaylist(name)

    monkeypatch.setattr(main, "get_plex_server", lambda: _DummyServer())

    main._run_playlists(main.playlists_data, completion_message="")

    assert processed == ["New Name"]
    assert deleted == ["Old Name"]

    with state_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved["playlists"] == ["New Name"]


def test_run_playlists_keeps_state_when_deletion_fails(tmp_path, monkeypatch):
    import main

    state_path = _prepare_state(tmp_path, ["Old Name"])
    processed = _install_common_patches(monkeypatch, main, state_path, {"New Name": {}})

    class _FailingPlaylist:
        def delete(self):
            raise RuntimeError("boom")

    class _DummyServer:
        def playlist(self, name):
            return _FailingPlaylist()

    monkeypatch.setattr(main, "get_plex_server", lambda: _DummyServer())

    main._run_playlists(main.playlists_data, completion_message="")

    assert processed == ["New Name"]

    with state_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)

    assert set(saved["playlists"]) == {"New Name", "Old Name"}


def test_run_playlists_drops_stale_state_when_playlist_missing(tmp_path, monkeypatch):
    import main

    class MissingPlaylistError(Exception):
        pass

    monkeypatch.setattr(main, "NotFound", MissingPlaylistError)

    state_path = _prepare_state(tmp_path, ["Old Name"])
    processed = _install_common_patches(monkeypatch, main, state_path, {"New Name": {}})

    class _DummyServer:
        def playlist(self, name):
            raise MissingPlaylistError()

    monkeypatch.setattr(main, "get_plex_server", lambda: _DummyServer())

    main._run_playlists(main.playlists_data, completion_message="")

    assert processed == ["New Name"]

    with state_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved["playlists"] == ["New Name"]
