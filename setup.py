#!/usr/bin/env python3
"""Create the virtual environment and install dependencies.

Cross-platform replacement for setup.sh / setup.ps1. Run with any Python 3:

    python setup.py        (or python3 setup.py)

Only the standard library is used, since dependencies are not installed yet.
"""

import os
import subprocess
import sys
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main() -> int:
    print(f"Creating virtual environment in {VENV_DIR}")
    venv.create(VENV_DIR, with_pip=True)

    python = venv_python(VENV_DIR)

    print("Installing Python dependencies")
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")],
        check=True,
    )

    print()
    print("Setup complete.")
    print("Run the app with: python serve.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
