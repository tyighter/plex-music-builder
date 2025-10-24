from datetime import timedelta
from typing import Any, Dict, Optional

import gui


class _TestableBuildManager(gui.BuildManager):
    """Test double that skips background log watcher threads."""

    def _start_log_watcher(self) -> None:  # type: ignore[override]
        self._log_watcher_thread = None


class _FakeProcess:
    def __init__(self) -> None:
        self._returncode: Optional[int] = None
        self.terminated = False
        self.killed = False
        self.wait_calls = 0
        self.stdout = None

    def poll(self) -> Optional[int]:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        if self._returncode is None:
            self._returncode = 0

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        del timeout
        self.wait_calls += 1
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def kill(self) -> None:
        self.killed = True
        self._returncode = 0


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


def test_stop_playlist_terminates_active_process() -> None:
    manager = _TestableBuildManager()
    fake_process = _FakeProcess()
    handle = gui._ProcessHandle(
        process=fake_process, job={"type": "playlist", "playlist": "Alpha"}
    )

    with manager._lock:
        manager._processes.append(handle)  # type: ignore[attr-defined]

    stopped, status, message = manager.stop_playlist("Alpha")

    assert stopped is True
    assert message == "Build for playlist 'Alpha' cancelled."
    assert status["running"] is False
    assert fake_process.terminated is True
    assert fake_process.wait_calls >= 1


def test_stop_playlist_removes_pending_job() -> None:
    manager = _TestableBuildManager()

    with manager._lock:
        manager._pending_jobs = [  # type: ignore[attr-defined]
            {"type": "playlist", "playlist": "Beta"}
        ]
        manager._queued_playlists = ["Beta"]  # type: ignore[attr-defined]

    stopped, status, message = manager.stop_playlist("Beta")

    assert stopped is True
    assert message == "Build for playlist 'Beta' cancelled."
    assert status["running"] is False
    assert manager._pending_jobs == []  # type: ignore[attr-defined]
    assert "Beta" not in status.get("waiting_playlists", [])


def test_stop_playlist_returns_error_when_missing() -> None:
    manager = _TestableBuildManager()

    stopped, status, message = manager.stop_playlist("Gamma")

    assert stopped is False
    assert status["running"] is False
    assert message == "No active build found for playlist 'Gamma'."
