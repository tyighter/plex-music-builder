import json
from pathlib import Path

import yaml


def _write_config(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def _prepare_app(tmp_path, monkeypatch, config_data):
    from gui import create_app

    config_path = tmp_path / "config.yml"
    _write_config(config_path, config_data)

    monkeypatch.setattr("gui.CONFIG_PATH", config_path)
    monkeypatch.setattr("gui.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("gui.RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("gui.PLAYLISTS_PATH", tmp_path / "playlists.yml")
    monkeypatch.setattr("gui.DEFAULT_ALLMUSIC_CACHE", tmp_path / "allmusic.json")
    monkeypatch.setattr(
        "gui.DEFAULT_LOG_PATH", tmp_path / "logs/plex_music_builder.log"
    )
    monkeypatch.setattr("gui._CONFIG_CACHE", {"mtime": None, "data": {}})
    monkeypatch.setattr("gui.CONFIG_SEARCH_PATHS", [config_path])

    app = create_app()
    app.testing = True
    return app, config_path


def _find_section(payload, section_id):
    sections = payload.get("sections") or []
    for section in sections:
        if isinstance(section, dict) and section.get("id") == section_id:
            return section
    return None


def _fields_to_map(section):
    mapping = {}
    for field in section.get("fields", []) or []:
        if isinstance(field, dict) and field.get("key"):
            mapping[field["key"]] = field
    return mapping


def test_get_config_endpoint_returns_sections(tmp_path, monkeypatch):
    config_data = {
        "plex": {
            "PLEX_URL": "http://localhost:32400",
            "PLEX_TOKEN": "token",
            "library_name": "Music",
        },
        "runtime": {"run_forever": True, "max_workers": 3},
        "logging": {"level": "INFO"},
        "allmusic": {"enabled": True, "timeout": 10},
    }

    app, config_path = _prepare_app(tmp_path, monkeypatch, config_data)
    client = app.test_client()

    response = client.get("/api/config")
    assert response.status_code == 200
    payload = response.get_json()

    assert payload["path"] == str(config_path)
    assert payload["supports_restart"] is True

    plex_section = _find_section(payload, "plex")
    assert plex_section is not None
    plex_fields = _fields_to_map(plex_section)
    assert plex_fields["PLEX_URL"]["value"] == "http://localhost:32400"
    assert plex_fields["library_name"]["value"] == "Music"

    runtime_section = _find_section(payload, "runtime")
    runtime_fields = _fields_to_map(runtime_section)
    assert runtime_fields["run_forever"]["value"] is True
    assert runtime_fields["max_workers"]["value"] == "3"


def test_update_config_endpoint_persists_changes(tmp_path, monkeypatch):
    config_data = {
        "plex": {
            "PLEX_URL": "http://old:32400",
            "PLEX_TOKEN": "old-token",
            "library_name": "Old",
        },
        "runtime": {"run_forever": True, "max_workers": 2},
        "logging": {"level": "INFO"},
    }

    app, config_path = _prepare_app(tmp_path, monkeypatch, config_data)
    client = app.test_client()

    payload = {
        "sections": {
            "plex": {
                "PLEX_URL": "http://new:32400",
                "PLEX_TOKEN": "new-token",
                "library_name": "New Library",
            },
            "runtime": {
                "run_forever": False,
                "max_workers": "5",
            },
            "logging": {"level": "DEBUG"},
        }
    }

    response = client.post(
        "/api/config", data=json.dumps(payload), content_type="application/json"
    )
    assert response.status_code == 200
    result = response.get_json()
    assert "message" in result

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["plex"]["PLEX_URL"] == "http://new:32400"
    assert saved["plex"]["PLEX_TOKEN"] == "new-token"
    assert saved["plex"]["library_name"] == "New Library"
    assert saved["runtime"]["run_forever"] is False
    assert saved["runtime"]["max_workers"] == 5
    assert saved["logging"]["level"] == "DEBUG"


def test_update_config_endpoint_rejects_invalid_numbers(tmp_path, monkeypatch):
    config_data = {"runtime": {"max_workers": 2}}
    app, _ = _prepare_app(tmp_path, monkeypatch, config_data)
    client = app.test_client()

    payload = {"sections": {"runtime": {"max_workers": "not-a-number"}}}
    response = client.post(
        "/api/config", data=json.dumps(payload), content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_restart_endpoint_requests_exit(tmp_path, monkeypatch):
    config_data = {}
    app, _ = _prepare_app(tmp_path, monkeypatch, config_data)

    triggered = []

    def fake_schedule(delay=1.0):
        triggered.append(delay)

    monkeypatch.setattr("gui._schedule_restart", fake_schedule)

    client = app.test_client()
    response = client.post("/api/runtime/restart")
    assert response.status_code == 202
    data = response.get_json()
    assert "message" in data
    assert triggered and triggered[0] >= 0
