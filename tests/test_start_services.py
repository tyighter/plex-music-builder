import contextlib
import io
import unittest
from typing import Iterable, List, Optional
from unittest.mock import patch

import start_services
from start_services import _coerce_runtime_flag


class FakeProcess:
    def __init__(self, poll_sequence: Iterable[Optional[int]], pid: int = 100) -> None:
        self._sequence: List[Optional[int]] = list(poll_sequence)
        if not self._sequence:
            self._sequence.append(None)
        self._index = 0
        self._returncode: Optional[int] = None
        self.pid = pid
        self.poll_calls = 0

    def poll(self) -> Optional[int]:
        if self._index < len(self._sequence):
            result = self._sequence[self._index]
            self._index += 1
        else:
            result = self._returncode

        self.poll_calls += 1
        if result is not None:
            self._returncode = result
        return result

    def terminate(self) -> None:
        self._returncode = self._returncode or 0

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:  # pragma: no cover - compatibility
        return self._returncode

    def kill(self) -> None:  # pragma: no cover - compatibility
        self._returncode = self._returncode or 0


class CoerceRuntimeFlagTests(unittest.TestCase):
    def test_truthy_values(self) -> None:
        truthy_values = [
            True,
            1,
            2,
            0.5,
            "true",
            "TRUE",
            " yes ",
            "Y",
            "on",
            "1",
        ]

        for value in truthy_values:
            with self.subTest(value=value):
                self.assertTrue(_coerce_runtime_flag(value))

    def test_falsy_values(self) -> None:
        falsy_values = [
            False,
            0,
            0.0,
            "false",
            "FALSE",
            " no ",
            "N",
            "off",
            "0",
            "",
            None,
        ]

        for value in falsy_values:
            with self.subTest(value=value):
                self.assertFalse(_coerce_runtime_flag(value))

    def test_unrecognised_strings_default_to_false(self) -> None:
        for value in ["maybe", "definitely", "null"]:
            with self.subTest(value=value):
                self.assertFalse(_coerce_runtime_flag(value))


class BuilderStartupDecisionTests(unittest.TestCase):
    def test_should_start_builder_when_flag_true(self) -> None:
        with patch(
            "start_services._load_runtime_config",
            return_value={"build_all_on_start": True},
        ):
            self.assertTrue(start_services._should_start_builder())

    def test_should_start_builder_when_flag_string_true(self) -> None:
        with patch(
            "start_services._load_runtime_config",
            return_value={"build_all_on_start": "true"},
        ):
            self.assertTrue(start_services._should_start_builder())

    def test_should_not_start_builder_when_flag_false(self) -> None:
        with patch(
            "start_services._load_runtime_config",
            return_value={"build_all_on_start": False},
        ):
            self.assertFalse(start_services._should_start_builder())

    def test_build_command_list_includes_builder_when_enabled(self) -> None:
        with patch(
            "start_services._load_runtime_config",
            return_value={"build_all_on_start": True},
        ):
            commands = start_services._build_command_list()

        builder_commands = [name for name, _ in commands if name == "builder"]
        self.assertEqual(builder_commands, ["builder"])

    def test_build_command_list_excludes_builder_when_disabled(self) -> None:
        with patch(
            "start_services._load_runtime_config",
            return_value={"build_all_on_start": "false"},
        ):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                commands = start_services._build_command_list()

        self.assertFalse(any(name == "builder" for name, _ in commands))
        self.assertIn("Build-on-start disabled", buffer.getvalue())


class SupervisorBehaviourTests(unittest.TestCase):
    def test_supervisor_keeps_gui_running_after_builder_finishes(self) -> None:
        builder_process = FakeProcess([0], pid=123)
        gui_process = FakeProcess([None, 0], pid=456)

        fake_commands = [("builder", ["builder"]), ("gui", ["gui"])]

        with patch("start_services.processes", []), patch(
            "start_services.COMMANDS", fake_commands
        ), patch("start_services.signal.signal"), patch(
            "start_services.time.sleep", side_effect=lambda _timeout: None
        ):

            def fake_start(name: str, command: List[str]) -> FakeProcess:
                process = builder_process if name == "builder" else gui_process
                start_services.processes.append((name, process))
                return process

            with patch("start_services.start_process", side_effect=fake_start):
                exit_code = start_services.supervise()

        self.assertEqual(exit_code, 0)
        self.assertEqual(builder_process.poll_calls, 1)
        self.assertGreaterEqual(gui_process.poll_calls, 2)


if __name__ == "__main__":  # pragma: no cover - unittest convenience
    unittest.main()
