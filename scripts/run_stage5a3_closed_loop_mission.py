from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import TextIO

WORLD_NAME = "construction_site_stage5a3_closed_loop_mission.wbt"
CONTROLLER_MARKERS = ("STAGE5A3_CONTROLLER_READY", "STAGE5A3_SUPERVISOR_READY")
WEBOTS_PROCESS_NAMES = ("webots.exe", "webotsw.exe", "webots-bin.exe")


def build_webots_arguments(world: Path, mode: str) -> list[str]:
    arguments = ["--mode=fast", "--stdout", "--stderr"]
    if mode == "validation":
        arguments.extend(("--batch", "--no-rendering"))
    arguments.append(str(world))
    return arguments


def validate_world_title(title: str) -> bool:
    normalized = title.casefold()
    return WORLD_NAME.casefold() in normalized and "empty.wbt" not in normalized


def visible_window_titles() -> list[tuple[int, str]]:
    if os.name != "nt":
        return []
    user32 = ctypes.windll.user32
    titles: list[tuple[int, str]] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def callback(hwnd: int, _: int) -> bool:
        length = user32.GetWindowTextLengthW(hwnd)
        if length:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            process_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            if buffer.value:
                titles.append((int(process_id.value), buffer.value))
        return True

    user32.EnumWindows(callback, 0)
    return titles


def stop_webots_processes() -> None:
    subprocess.run(
        ["taskkill", "/F", "/T", "/IM", "webots.exe"],
        check=False,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["taskkill", "/F", "/T", "/IM", "webotsw.exe"],
        check=False,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["taskkill", "/F", "/T", "/IM", "webots-bin.exe"],
        check=False,
        capture_output=True,
        text=True,
    )


def read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def stream_output(process: subprocess.Popen[str], log_handle: TextIO) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        log_handle.write(line)
        log_handle.flush()


def run(project: Path, mode: str, timeout: int, launch_check_only: bool = False) -> int:
    world = project / "webots" / "worlds" / WORLD_NAME
    output = project / "webots" / "logs" / "stage5a3_closed_loop"
    python = project / ".venv" / "Scripts" / "python.exe"
    webots_home = Path(os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots"))
    webots_bin = webots_home / "msys64" / "mingw64" / "bin"
    executable = webots_bin / "webots.exe"
    validator = project / "scripts" / "validate_stage5a3_closed_loop_mission.py"
    required = (world, python, executable, validator)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required runtime files are missing: {missing}")

    stop_webots_processes()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    runtime_log = output / "webots_runtime.log"
    environment = os.environ.copy()
    environment.update(
        {
            "WEBOTS_HOME": str(webots_home),
            "WEBOTS_PYTHON_COMMAND": str(python),
            "RISK_AWARE_PROJECT_ROOT": str(project),
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": str(webots_home / "lib" / "controller" / "python"),
            "PATH": os.pathsep.join((str(python.parent), str(webots_bin), environment["PATH"])),
        }
    )
    command = [str(executable), *build_webots_arguments(world, mode)]
    launch_record = {
        "command": command,
        "mode": mode,
        "world": str(world),
        "started_at_unix": time.time(),
        "launch_check_only": launch_check_only,
    }
    (output / "launcher_record.json").write_text(
        json.dumps(launch_record, indent=2), encoding="utf-8"
    )

    with runtime_log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=project,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        output_thread = threading.Thread(
            target=stream_output,
            args=(process, log_handle),
            daemon=True,
        )
        output_thread.start()
        launch_record["starter_process_id"] = process.pid
        controller_deadline = time.monotonic() + 45
        mission_deadline = time.monotonic() + timeout
        title_verified = False
        markers_verified = False
        try:
            while time.monotonic() < mission_deadline:
                titles = visible_window_titles()
                wrong_titles = [title for _, title in titles if "empty.wbt" in title.casefold()]
                if wrong_titles:
                    raise RuntimeError(f"Webots opened the wrong world: {wrong_titles}")
                matching = [(pid, title) for pid, title in titles if validate_world_title(title)]
                if matching:
                    title_verified = True
                    launch_record["main_process_id"] = matching[0][0]
                    launch_record["verified_window_title"] = matching[0][1]

                log_text = read_log(runtime_log)
                markers_verified = all(marker in log_text for marker in CONTROLLER_MARKERS)
                launch_record["title_verified"] = title_verified
                launch_record["controller_markers_verified"] = markers_verified
                (output / "launcher_record.json").write_text(
                    json.dumps(launch_record, indent=2), encoding="utf-8"
                )
                if time.monotonic() >= controller_deadline and not markers_verified:
                    raise RuntimeError(
                        "Controller startup markers did not appear within 45 seconds."
                    )
                if (
                    mode == "interactive"
                    and time.monotonic() >= controller_deadline
                    and not title_verified
                ):
                    raise RuntimeError("The Stage 5A3 Webots window title was not verified.")
                if launch_check_only and title_verified and markers_verified:
                    launch_record["launch_check_passed"] = True
                    break
                if (output / "stage5a3_failure.marker").exists():
                    raise RuntimeError("The Stage 5A3 controller reported a runtime failure.")
                if (output / "stage5a3_complete.marker").exists():
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"Webots exited early with code {process.returncode}.")
                time.sleep(0.25)
            else:
                raise RuntimeError(f"Stage 5A3 exceeded the {timeout}-second launcher timeout.")
        finally:
            launch_record["title_verified"] = title_verified
            launch_record["controller_markers_verified"] = markers_verified
            (output / "launcher_record.json").write_text(
                json.dumps(launch_record, indent=2), encoding="utf-8"
            )
            if mode == "validation" or launch_check_only:
                stop_webots_processes()

    if launch_check_only:
        return 0

    result = subprocess.run([str(python), str(validator), str(project)], check=False)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--mode", choices=("interactive", "validation"), default="validation")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--launch-check-only", action="store_true")
    arguments = parser.parse_args()
    try:
        raise SystemExit(
            run(
                arguments.project.resolve(),
                arguments.mode,
                arguments.timeout,
                arguments.launch_check_only,
            )
        )
    except Exception as exception:
        print(f"Stage 5A3 launcher failure: {exception}", file=sys.stderr)
        stop_webots_processes()
        raise SystemExit(1) from exception


if __name__ == "__main__":
    main()
