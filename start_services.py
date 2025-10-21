from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

ProcessInfo = Tuple[str, subprocess.Popen]


BASE_DIR = Path(__file__).resolve().parent


def _resolve_config_path() -> Path:
    candidates = []

    env_override = os.environ.get("PMB_CONFIG_PATH")
    if env_override:
        candidates.append(Path(env_override).expanduser())

    candidates.append(Path("/app/config.yml"))
    candidates.append(BASE_DIR / "config.yml")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


CONFIG_PATH = _resolve_config_path()


def _load_runtime_config() -> Dict[str, object]:
    if not CONFIG_PATH.exists():
        return {}

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config_data = yaml.safe_load(config_file) or {}
    except Exception:
        return {}

    runtime_cfg = config_data.get("runtime")
    return runtime_cfg if isinstance(runtime_cfg, dict) else {}


def _should_start_builder() -> bool:
    """Return ``True`` when the builder should auto-run on service startup."""
    runtime_cfg = _load_runtime_config()
    return bool(runtime_cfg.get("build_all_on_start", False))


def _build_command_list() -> List[Tuple[str, List[str]]]:
    commands: List[Tuple[str, List[str]]] = []

    if _should_start_builder():
        commands.append(("builder", [sys.executable, "main.py"]))
    else:
        print(
            "Build-on-start disabled via runtime.build_all_on_start; builder will remain idle until triggered.",
            flush=True,
        )

    commands.append(("gui", [sys.executable, "gui.py"]))
    return commands


COMMANDS: List[Tuple[str, List[str]]] = _build_command_list()

processes: List[ProcessInfo] = []

def start_process(name: str, command: List[str]) -> subprocess.Popen:
    print(f"Starting {name}: {' '.join(command)}", flush=True)
    process = subprocess.Popen(command)
    processes.append((name, process))
    return process

def terminate_all() -> None:
    deadline = time.time() + 10
    for name, process in processes:
        if process.poll() is None:
            print(f"Stopping {name} (pid={process.pid})", flush=True)
            process.terminate()

    for name, process in processes:
        if process.poll() is None:
            remaining = max(0.0, deadline - time.time())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"Force killing {name} (pid={process.pid})", flush=True)
                process.kill()

    for _, process in processes:
        try:
            process.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            process.kill()


def handle_signal(signum: int, frame: object) -> None:  # pragma: no cover - signal handler
    print(f"Received signal {signum}. Shutting down services...", flush=True)
    terminate_all()
    sys.exit(0)


def supervise() -> int:
    exit_code = 0

    for name, command in COMMANDS:
        start_process(name, command)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_signal)

    try:
        while True:
            for name, process in processes:
                return_code = process.poll()
                if return_code is not None:
                    if return_code != 0:
                        exit_code = return_code
                        print(
                            f"Process '{name}' exited with code {return_code}.", flush=True
                        )
                    else:
                        print(f"Process '{name}' completed successfully.", flush=True)
                    return exit_code
            time.sleep(1)
    except KeyboardInterrupt:  # pragma: no cover - manual interrupt
        print("Keyboard interrupt received. Shutting down...", flush=True)
    finally:
        terminate_all()

    return exit_code


if __name__ == "__main__":
    sys.exit(supervise())
