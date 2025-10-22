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
