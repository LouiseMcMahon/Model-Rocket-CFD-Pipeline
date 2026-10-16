"""Run OpenFOAM commands, either natively on Linux or inside WSL from Windows.

Cases always live on a Linux filesystem (paths in this module are POSIX strings). From
Windows they are read and written through the \\\\wsl.localhost share, which keeps the
case paths free of spaces and avoids slow /mnt/c I/O while the solver runs.
"""

from __future__ import annotations

import logging
import platform
import shlex
import subprocess
from pathlib import Path, PurePosixPath

from .config import OpenFOAMConfig

log = logging.getLogger(__name__)


class OpenFOAMError(RuntimeError):
    pass


class Runner:
    def __init__(self, cfg: OpenFOAMConfig):
        self.cfg = cfg
        mode = cfg.runner
        if mode == "auto":
            mode = "wsl" if platform.system() == "Windows" else "native"
        if mode not in ("wsl", "native"):
            raise ValueError(f"openfoam.runner must be auto, wsl or native, not '{mode}'")
        self.mode = mode
        self.work_dir = PurePosixPath(self._expand_home(cfg.work_dir))

    # -- paths -------------------------------------------------------------------------

    def _expand_home(self, path: str) -> str:
        if not path.startswith("~"):
            return path
        home = self.shell("printf %s \"$HOME\"").strip()
        return home + path[1:]

    def local(self, linux_path: PurePosixPath | str) -> Path:
        """Path that Python on this machine can open for a Linux-side path."""
        p = str(linux_path)
        if self.mode == "native":
            return Path(p)
        if p.startswith("/mnt/") and len(p) > 6 and p[6] in "/":
            return Path(f"{p[5].upper()}:/{p[7:]}")
        return Path(f"//wsl.localhost/{self.cfg.wsl_distro}{p}")

    # -- execution ---------------------------------------------------------------------

    def _argv(self) -> list[str]:
        if self.mode == "native":
            return ["bash", "-s"]
        return ["wsl.exe", "-d", self.cfg.wsl_distro, "--exec", "bash", "-s"]

    def shell(self, script: str) -> str:
        res = subprocess.run(self._argv(), input=script.encode(), capture_output=True)
        if res.returncode:
            raise OpenFOAMError(f"Shell command failed: {script}\n{res.stderr.decode(errors='replace')}")
        return res.stdout.decode(errors="replace")

    def check(self) -> str:
        """Verify OpenFOAM can be sourced; returns its version string."""
        out = self.shell(f"source {shlex.quote(self.cfg.bashrc)} >/dev/null 2>&1; "
                            "command -v simpleFoam >/dev/null || exit 3; echo ${WM_PROJECT_VERSION:-unknown}")
        return out.strip()

    def run(self, case: PurePosixPath, commands: list[tuple[str, str]]) -> None:
        """Run (log_name, command) pairs in a case directory, stopping at the first failure.

        Each command's output goes to ``<case>/log.<log_name>``.
        """
        lines = ["#!/bin/bash", f"source {shlex.quote(self.cfg.bashrc)} >/dev/null 2>&1",
                 f"cd {shlex.quote(str(case))} || exit 90"]
        for i, (name, cmd) in enumerate(commands):
            # stdin from /dev/null: mpirun would otherwise forward (and swallow) our stdin.
            lines.append(f"{cmd} < /dev/null > log.{name} 2>&1 || {{ echo FAILED:{name}; exit {100 + i}; }}")
        script_name = "rocketcfd_run.sh"
        write_text(self.local(case / script_name), "\n".join(lines) + "\n")
        log.debug("Running %s in %s", script_name, case)
        proc = subprocess.run(self._argv(), input=f"bash {shlex.quote(str(case / script_name))}\n".encode(),
                              capture_output=True)
        out = proc.stdout.decode(errors="replace")
        if proc.returncode:
            failed = next((l.split(":", 1)[1] for l in out.splitlines() if l.startswith("FAILED:")), None)
            if failed is None and 100 <= proc.returncode < 100 + len(commands):
                failed = commands[proc.returncode - 100][0]
            tail = ""
            if failed:
                logfile = self.local(case / f"log.{failed}")
                if logfile.exists():
                    tail = "\n".join(logfile.read_text(errors="replace").splitlines()[-40:])
            raise OpenFOAMError(f"OpenFOAM step '{failed or '?'}' failed in {case} "
                                f"(exit {proc.returncode}).\n{proc.stderr.decode(errors='replace')}\n{tail}")


def write_text(path: Path, text: str) -> None:
    """Write a file with Unix line endings (OpenFOAM and bash both need them)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
