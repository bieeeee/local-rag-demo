#!/usr/bin/env python3
"""Start the web server and open the browser once it is ready.

Cross-platform replacement for serve.sh / serve.ps1. Run with any Python 3:

    python serve.py        (or python3 serve.py)

The launcher re-executes itself inside .venv so the correct interpreter is used,
then runs uvicorn and opens the browser as soon as the port responds.
"""

import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
URL = "http://127.0.0.1:8000"
HOST = "127.0.0.1"
PORT = 8000


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def reexec_in_venv() -> None:
    """Re-run this script with the venv interpreter if not already using it."""
    # If we are already running inside .venv, nothing to do. Comparing sys.prefix
    # avoids following the venv's python symlink, which resolves back to the base
    # interpreter and would make a path comparison unreliable.
    if Path(sys.prefix).resolve() == VENV_DIR.resolve():
        return

    python = venv_python(VENV_DIR)
    if not python.exists():
        sys.exit("Virtual environment not found. Run 'python setup.py' first.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit(result.returncode)


def open_browser_when_ready() -> None:
    # uvicorn only binds the port after lifespan startup (embedding / DB load)
    # finishes, so poll until the URL responds, then open the browser once.
    # The first run can take a while because the model may need downloading.
    for _ in range(600):
        try:
            with urllib.request.urlopen(URL, timeout=2) as response:
                response.read(1)
            webbrowser.open(URL)
            return
        except Exception:
            time.sleep(1)


def main() -> int:
    reexec_in_venv()

    # uvicorn is available now that we are inside the venv.
    import uvicorn

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    uvicorn.run("server:app", host=HOST, port=PORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
