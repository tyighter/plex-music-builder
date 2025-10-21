import contextlib
import io
import unittest
from unittest.mock import patch

import start_services
from start_services import _coerce_runtime_flag


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


if __name__ == "__main__":  # pragma: no cover - unittest convenience
    unittest.main()
