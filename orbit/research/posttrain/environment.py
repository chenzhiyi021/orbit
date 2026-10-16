"""Record exact installed packages and source bytes; verify the common M6 anchor."""

from __future__ import annotations
import importlib.metadata as metadata
from pathlib import Path
import platform
import subprocess
import sys
from .protocol import DEPENDENCIES, digest


def inspect_environment(source_roots, *, check=True):
    rows = {}
    errors = []
    for name, expected in DEPENDENCIES.items():
        root = Path(source_roots.get(name, "")).resolve()
        p = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        head = p.stdout.strip() if p.returncode == 0 else None
        if head != expected:
            errors.append(f"{name}: expected {expected}, found {head} at {root}")
        sources = {str(f.relative_to(root)): digest(f) for f in root.rglob("*.py") if ".git" not in f.parts and not any(part.startswith(".venv") for part in f.parts)} if head else {}
        rows[name] = {"root": str(root), "head": head, "source_sha256": sources, "diff": subprocess.run(["git", "-C", str(root), "diff", "HEAD"], capture_output=True, text=True).stdout if head else None}
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata.get("Name")}
    if check:
        import torch

        if torch.__version__ != "2.11.0+cu130":
            errors.append(f"Expected audited torch2.11.0+cu130, found {torch.__version__}")
        if sys.version_info[:2] != (3, 12):
            errors.append("Runtime profile requires Python3.12")
    result = {"python": sys.version, "platform": platform.platform(), "packages": packages, "sources": rows, "errors": errors, "verified": not errors}
    if check and errors:
        raise ValueError("Common environment anchor failed:\n" + "\n".join(errors))
    return result
