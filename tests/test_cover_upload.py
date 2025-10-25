from io import BytesIO
def _create_test_app(tmp_path, monkeypatch):
    from gui import create_app

    config_path = tmp_path / "config.yml"
    config_path.write_text("{}", encoding="utf-8")

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
    return app


def test_upload_cover_saves_file_and_returns_relative_path(tmp_path, monkeypatch):
    upload_dir = tmp_path / "covers"
    monkeypatch.setenv("PMB_COVER_UPLOAD_DIR", str(upload_dir))

    app = _create_test_app(tmp_path, monkeypatch)
    client = app.test_client()

    image_bytes = b"\x89PNG\r\n\x1a\nminimal"
    data = {
        "file": (BytesIO(image_bytes), "artwork.PNG"),
    }

    response = client.post(
        "/api/upload_cover",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["status"] == "uploaded"
    assert payload["path"] == "covers/artwork.png"

    saved_file = upload_dir / "artwork.png"
    assert saved_file.exists()
    assert saved_file.read_bytes() == image_bytes


def test_upload_cover_rejects_non_image_extension(tmp_path, monkeypatch):
    upload_dir = tmp_path / "covers"
    monkeypatch.setenv("PMB_COVER_UPLOAD_DIR", str(upload_dir))

    app = _create_test_app(tmp_path, monkeypatch)
    client = app.test_client()

    data = {
        "file": (BytesIO(b"not-an-image"), "notes.txt"),
    }

    response = client.post(
        "/api/upload_cover",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert "error" in payload
    assert not (upload_dir / "notes.txt").exists()


def test_upload_cover_requires_file(tmp_path, monkeypatch):
    upload_dir = tmp_path / "covers"
    monkeypatch.setenv("PMB_COVER_UPLOAD_DIR", str(upload_dir))

    app = _create_test_app(tmp_path, monkeypatch)
    client = app.test_client()

    response = client.post(
        "/api/upload_cover",
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert "error" in payload
