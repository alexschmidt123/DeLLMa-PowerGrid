#!/usr/bin/env python3
"""Start the Simulink workflow: ``overwrite_ieee14_simulink.m`` + ``intial_settings.json`` via ``matlab -batch``."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_matlab() -> str | None:
    exe = shutil.which("matlab")
    if exe:
        return exe
    apps = Path("/Applications")
    if apps.is_dir():
        for name in sorted(apps.iterdir()):
            if name.name.startswith("MATLAB_R") and name.name.endswith(".app"):
                p = name / "bin" / "matlab"
                if p.is_file():
                    return str(p)
    return None


def main() -> None:
    root = Path(__file__).resolve().parent
    matlab = os.environ.get("MATLAB_BIN") or find_matlab()
    if not matlab:
        print("error: matlab executable not found (set MATLAB_BIN or add to PATH)", file=sys.stderr)
        sys.exit(1)

    mdir = str(root).replace("'", "''")
    cmd = (
        f"cd('{mdir}'); "
        "overwrite_ieee14_simulink; "
        "exit;"
    )
    try:
        subprocess.run([matlab, "-batch", cmd], check=True, env=os.environ.copy())
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"error: MATLAB run failed: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
