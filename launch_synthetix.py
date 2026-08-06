#!/usr/bin/env python3
"""One-click local launcher for Synthetix on Windows.

Responsibilities:
- run from any working directory (repository root is derived from this file)
- create/use ``.venv`` with a Python 3.10+ interpreter
- install project dependencies only when the import gate fails
- launch the selected detector backend via ``server.py``
- poll ``/health`` until the server is ready
- open http://localhost:8000 only after readiness
- keep the console open for logs and shut the server down cleanly on exit

Diagnostic environment variables (not part of the product flow):
- SYNTETIX_LAUNCH_NO_VENV=1  -> use the current interpreter instead of .venv
- SYNTETIX_NO_BROWSER=1      -> do not open the browser (automated checks)
"""

import argparse
import atexit
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SERVER_SCRIPT = REPO_ROOT / "server.py"
VENV_DIR = REPO_ROOT / ".venv"

BACKEND_LABELS = {
    "hc3_roberta": "Fast Baseline — faster, but frequently misses unfamiliar AI writing",
    "desklib_academic": "Academic Sensitive — high recall, high false-positive risk",
    "balanced_review": "Experimental Comparison — two detectors with an uncertain outcome when they disagree",
}

DEFAULT_PORTS = {
    "hc3_roberta": 8000,
    "desklib_academic": 8001,
    "balanced_review": 8002,
}

REQUIRED_IMPORTS = ["fastapi", "uvicorn", "torch", "transformers", "sklearn", "requests"]
DEFAULT_HEALTH_TIMEOUT_S = 900


def backend_label(backend: str) -> str:
    return BACKEND_LABELS.get(backend, backend)


def health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"


def python_major_minor(python_cmd: list):
    try:
        out = subprocess.run(
            python_cmd + ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0:
            major, minor = out.stdout.strip().split(".")
            return int(major), int(minor)
    except Exception:
        pass
    return None


def find_system_python() -> list:
    """Return a command (list) for a usable Python 3.10+ interpreter."""
    candidates = [[sys.executable]]
    py_launcher = shutil.which("py")
    if py_launcher:
        for ver in ("-3.12", "-3.11", "-3.10", "-3"):
            candidates.append([py_launcher, ver])
    for name in ("python", "python3"):
        p = shutil.which(name)
        if p:
            candidates.append([p])
    # Probe common Windows install locations when nothing is on PATH.
    roots = [
        os.environ.get("LOCALAPPDATA", ""),
        os.environ.get("ProgramFiles", ""),
        os.environ.get("ProgramFiles(x86)", ""),
    ]
    for root in roots:
        if not root:
            continue
        for sub in (
            "Programs\\Python\\Python313",
            "Programs\\Python\\Python312",
            "Programs\\Python\\Python311",
            "Programs\\Python\\Python310",
            "Python313",
            "Python312",
            "Python311",
            "Python310",
            "Python313-32",
            "Python312-32",
            "Python311-32",
            "Python310-32",
        ):
            exe = os.path.join(root, sub, "python.exe")
            if os.path.exists(exe):
                candidates.append([exe])
    seen = set()
    for cand in candidates:
        key = " ".join(cand)
        if key in seen:
            continue
        seen.add(key)
        mm = python_major_minor(cand)
        if mm and mm >= (3, 10):
            return cand
    return []


def venv_python() -> Path:
    return VENV_DIR / "Scripts" / "python.exe"


def deps_ok(python_cmd: list) -> bool:
    code = (
        "import fastapi, uvicorn, torch, transformers, sklearn, requests; "
        "print('deps ok')"
    )
    try:
        out = subprocess.run(
            python_cmd + ["-c", code], capture_output=True, text=True, timeout=120
        )
        return out.returncode == 0
    except Exception:
        return False


def install_deps(python_cmd: list) -> bool:
    print("Installing project dependencies (first launch; this can take several minutes)...")
    pip = python_cmd + ["-m", "pip"]
    if subprocess.run(pip + ["install", "--upgrade", "pip"], cwd=REPO_ROOT).returncode != 0:
        print("WARNING: could not upgrade pip; continuing with existing pip.")
    result = subprocess.run(
        pip + ["install", "-e", ".[dev,benchmark]"], cwd=REPO_ROOT
    )
    return result.returncode == 0


def wait_for_ready(port: int, backend: str, timeout_s: int = DEFAULT_HEALTH_TIMEOUT_S):
    """Poll /health until model_loaded is true and backend_name matches.

    Returns (ok, elapsed_s, health)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        health = probe_health(port)
        if backend_ready(health, backend):
            return True, time.time() - t0, health
        time.sleep(1.0)
    return False, time.time() - t0, probe_health(port)


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        pass


def _enable_job_kill_on_close(proc):
    """Ensure the server process dies with the launcher (Windows job object)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        )
        if ok:
            kernel32.AssignProcessToJobObject(job, ctypes.c_void_p(int(proc._handle)))
    except Exception:
        pass


def install_console_close_handler(proc_holder):
    """Catch console close/ctrl events on Windows and stop the server first."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
        def handler(ctrl_type):
            stop_server(proc_holder[0])
            return True

        ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, True)
    except Exception:
        pass


def probe_health(port: int):
    """Return parsed /health JSON for the port, or None if unreachable/not JSON."""
    try:
        with urllib.request.urlopen(health_url(port), timeout=3) as res:
            if res.status != 200:
                return None
            import json
            return json.loads(res.read().decode("utf-8"))
    except Exception:
        return None


def backend_ready(health: dict, backend: str) -> bool:
    """True only when /health reports model_loaded and the exact backend name."""
    return bool(health) and health.get("model_loaded") is True and health.get("backend_name") == backend


def classify_existing(health, backend: str) -> str:
    """Decide what to do with a server already answering on the port.

    Returns 'reuse' (same backend), 'conflict' (different Synthetix backend),
    or 'launch' (no server / unrelated response). A different backend is never
    silently reused.
    """
    if health is None:
        return "launch"
    running = health.get("backend_name")
    if running == backend:
        return "reuse"
    if running:
        return "conflict"
    return "launch"


def resolve_port(backend: str, requested: int) -> int:
    return requested or DEFAULT_PORTS.get(backend, 8000)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Synthetix locally")
    parser.add_argument(
        "--backend",
        choices=sorted(BACKEND_LABELS),
        default=os.environ.get("SYNTETIX_BACKEND", "hc3_roberta"),
        help="Detector backend to launch (default: hc3_roberta)",
    )
    parser.add_argument("--port", type=int, default=None, help="Port override (defaults are per backend: 8000 hc3, 8001 desklib, 8002 balanced)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser")
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=DEFAULT_HEALTH_TIMEOUT_S,
        help="Seconds to wait for /health before giving up",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("Synthetix — local writing-integrity signal tool")
    print(f"Backend: {args.backend}")
    print(f"Mode:    {backend_label(args.backend)}")
    print("Results are experimental and must not be used as proof of misconduct.")
    print("=" * 64)

    port = resolve_port(args.backend, args.port)
    existing = probe_health(port)
    decision = classify_existing(existing, args.backend)
    if decision == "reuse":
        print(f"An existing Synthetix server with backend '{args.backend}' is already running on port {port}.")
        if not backend_ready(existing, args.backend):
            print("Waiting for the existing server's detector to finish loading...")
            ok, elapsed, health = wait_for_ready(port, args.backend, args.health_timeout)
            if not ok:
                print("ERROR: the existing server did not become ready in time.", file=sys.stderr)
                return 1
            print(f"Detector ready after {elapsed:.0f}s.")
        else:
            print("Detector ready.")
        if not args.no_browser and os.environ.get("SYNTETIX_NO_BROWSER") != "1":
            webbrowser.open(f"http://localhost:{port}?expected_backend={args.backend}")
        else:
            print(f"Open http://localhost:{port}?expected_backend={args.backend} in your browser.")
        return 0
    if decision == "conflict":
        running_backend = (existing or {}).get("backend_name")
        print(
            f"ERROR: port {port} is already running Synthetix with backend "
            f"'{running_backend or 'unknown'}' but '{args.backend}' was requested.",
            file=sys.stderr,
        )
        print(
            "Close that server first, then relaunch. Each backend has a dedicated "
            "port: hc3_roberta=8000, desklib_academic=8001, balanced_review=8002.",
            file=sys.stderr,
        )
        return 1

    use_current = os.environ.get("SYNTETIX_LAUNCH_NO_VENV") == "1"
    if use_current:
        python_cmd = [sys.executable]
        mm = python_major_minor(python_cmd)
        if not mm or mm < (3, 10):
            print(f"ERROR: current Python is not 3.10+ ({sys.executable}).", file=sys.stderr)
            return 1
        print(f"Using current Python: {sys.executable}")
    else:
        python_cmd = find_system_python()
        if not python_cmd:
            print(
                "ERROR: no Python 3.10+ installation found. Install Python 3.10-3.12 "
                "from https://www.python.org/downloads/ and try again.",
                file=sys.stderr,
            )
            return 1
        print(f"Using Python: {' '.join(python_cmd)}")

        if VENV_DIR.exists() and venv_python().exists():
            print("Virtual environment found at .venv")
        else:
            print("Creating virtual environment (.venv)...")
            result = subprocess.run(
                python_cmd + ["-m", "venv", str(VENV_DIR)], cwd=REPO_ROOT
            )
            if result.returncode != 0:
                print("ERROR: could not create the virtual environment.", file=sys.stderr)
                return 1
        python_cmd = [str(venv_python())]

        if deps_ok(python_cmd):
            print("Dependencies already installed; skipping installation.")
        else:
            if not install_deps(python_cmd):
                print(
                    "ERROR: dependency installation failed. Check the messages above and "
                    "your network connection.",
                    file=sys.stderr,
                )
                return 1

    print(f"Launching backend '{args.backend}' on port {port}...")
    env = os.environ.copy()
    env["SYNTETIX_BACKEND"] = args.backend
    proc = subprocess.Popen(
        python_cmd + [
            str(SERVER_SCRIPT),
            "--backend",
            args.backend,
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    proc_holder = [proc]
    atexit.register(stop_server, proc)
    _enable_job_kill_on_close(proc)
    install_console_close_handler(proc_holder)

    print("Waiting for the detector to load...")
    print("  (first launch may download model files and take a while)")
    ok, elapsed, health = wait_for_ready(port, args.backend, args.health_timeout)
    if not ok:
        reported_backend = (health or {}).get("backend_name")
        if reported_backend and reported_backend != args.backend:
            print(
                f"ERROR: startup failed because /health reports backend "
                f"'{reported_backend}' but '{args.backend}' was requested.",
                file=sys.stderr,
            )
        else:
            print(
                "ERROR: the server did not become ready in time. This is usually a model "
                "download failure or insufficient memory (the detector needs ~2 GB).",
                file=sys.stderr,
            )
        if proc.poll() is not None:
            print(
                "  The server process exited on its own; see its output above for the "
                "technical cause.",
                file=sys.stderr,
            )
        stop_server(proc)
        return 1

    print(f"Detector ready (health OK after {elapsed:.0f}s).")
    url = f"http://localhost:{port}?expected_backend={args.backend}"
    if not args.no_browser and os.environ.get("SYNTETIX_NO_BROWSER") != "1":
        print(f"Opening {url} in your browser...")
        webbrowser.open(url)
    else:
        print(f"Open {url} in your browser.")

    print("")
    print("Server is running. Press Ctrl+C (or close this window) to stop it.")
    try:
        while True:
            if proc.poll() is not None:
                print(
                    "WARNING: the server process exited unexpectedly. See its output above.",
                    file=sys.stderr,
                )
                return 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping Synthetix...")
    finally:
        stop_server(proc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
