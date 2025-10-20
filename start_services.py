from __future__ import annotations

import signal
import subprocess
import sys
import time
from typing import List, Tuple

ProcessInfo = Tuple[str, subprocess.Popen]
COMMANDS: List[Tuple[str, List[str]]] = [
    ("builder", [sys.executable, "main.py"]),
    ("gui", [sys.executable, "gui.py"]),
]

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
