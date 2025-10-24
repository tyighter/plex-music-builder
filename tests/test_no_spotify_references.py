from __future__ import annotations

from pathlib import Path


CURRENT_TEST_FILE = Path(__file__).resolve()


def _iter_repo_files(root: Path):
    allowed_suffixes = {
        ".py",
        ".html",
        ".js",
        ".css",
        ".json",
        ".yml",
        ".yaml",
        ".txt",
        ".cfg",
        ".ini",
        ".env",
    }
    allowed_names = {
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        "config",
    }

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if "__pycache__" in rel_parts:
            continue

        if path.suffix and path.suffix.lower() in allowed_suffixes:
            yield path
            continue

        if path.name in allowed_names:
            yield path


def test_no_spotify_mentions():
    repo_root = Path(__file__).resolve().parents[1]
    offending_files = []

    for file_path in _iter_repo_files(repo_root):
        if file_path.resolve() == CURRENT_TEST_FILE:
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

        if "spotify" in content.lower():
            offending_files.append(str(file_path.relative_to(repo_root)))

    assert not offending_files, (
        "Expected no references to Spotify in the repository, but found mentions in: "
        + ", ".join(offending_files)
    )
