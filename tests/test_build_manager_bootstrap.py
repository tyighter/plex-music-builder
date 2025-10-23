from datetime import timedelta
from typing import Any, Dict

import gui


class _TestableBuildManager(gui.BuildManager):
    """Test double that skips background log watcher threads."""

    def _start_log_watcher(self) -> None:  # type: ignore[override]
        self._log_watcher_thread = None


def _build_manager_status(manager: gui.BuildManager) -> Dict[str, Any]:
    """Helper to safely fetch a status snapshot in tests."""

    return manager.get_status()


def test_bootstrap_log_lines_do_not_mark_as_running() -> None:
    manager = _TestableBuildManager()

    manager._handle_log_line(
        "[INFO] Processing 2 playlist(s): Alpha, Beta",
        is_bootstrap=True,
    )
    manager._handle_log_line(
        "[INFO] Build started for playlist 'Alpha'",
        is_bootstrap=True,
    )

    status = _build_manager_status(manager)

    assert status["running"] is False
    assert status["active_playlists"] == []
    assert status["waiting_playlists"] == []
    assert manager._queued_playlists == []  # type: ignore[attr-defined]
    assert status["message"] == "Build started for playlist 'Alpha'"


def test_non_bootstrap_log_line_marks_running() -> None:
    manager = _TestableBuildManager()

    manager._handle_log_line("[INFO] Build started for playlist 'Gamma'")

    status = _build_manager_status(manager)

    assert status["running"] is True
    assert status["active_playlists"] == ["Gamma"]
    assert status["waiting_playlists"] == []


def test_completed_playlist_logs_expire_after_retention() -> None:
    manager = _TestableBuildManager()

    manager._handle_log_line("[INFO] Build started for playlist 'Alpha'")
    manager._handle_log_line("[INFO] ✅ Finished building 'Alpha' (10 tracks)")

    initial_status = _build_manager_status(manager)
    assert "Alpha" in initial_status["logs"]["playlists"]

    with manager._lock:
        manager._playlist_completed_at["Alpha"] = (
            gui._utcnow() - gui.PLAYLIST_ACTIVITY_RETENTION - timedelta(seconds=1)
        )

    expired_status = _build_manager_status(manager)
    assert "Alpha" not in expired_status["logs"]["playlists"]
