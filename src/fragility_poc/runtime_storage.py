"""Keep all program-owned runtime writes inside the research project."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def _inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def configure_runtime_storage(
    project_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    """Route temporary files and library caches to the project drive.

    The batch runs for days and can create many OpenSees recorder files and
    joblib/matplotlib cache entries.  Windows normally places these under the
    user's C-drive profile.  The research workstation instead keeps every
    program-owned writable runtime location under ``Demo Program`` so Drive C
    is not consumed.
    """
    project = (
        Path(project_root).absolute()
        if project_root is not None
        else Path(__file__).absolute().parents[2]
    )
    configured = runtime_root or os.environ.get(
        "FRAGILITY_POC_RUNTIME_ROOT"
    )
    storage = (
        Path(configured).absolute()
        if configured
        else (project / "runtime_storage").absolute()
    )
    if storage.drive.upper() == "C:":
        raise RuntimeError(
            "Runtime storage must not use Drive C on this workstation"
        )
    if not _inside(storage, project):
        raise RuntimeError(
            "Runtime storage must remain inside the Demo Program project"
        )

    directories = {
        "root": storage,
        "temp": storage / "tmp",
        "matplotlib": storage / "cache" / "matplotlib",
        "joblib": storage / "cache" / "joblib",
        "xdg": storage / "cache" / "xdg",
        "pip": storage / "cache" / "pip",
        "node": storage / "cache" / "node",
        "numba": storage / "cache" / "numba",
        "torch": storage / "cache" / "torch",
        "jupyter": storage / "jupyter",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    environment = {
        "FRAGILITY_POC_RUNTIME_ROOT": directories["root"],
        "TEMP": directories["temp"],
        "TMP": directories["temp"],
        "TMPDIR": directories["temp"],
        "MPLCONFIGDIR": directories["matplotlib"],
        "JOBLIB_TEMP_FOLDER": directories["joblib"],
        "XDG_CACHE_HOME": directories["xdg"],
        "PIP_CACHE_DIR": directories["pip"],
        "npm_config_cache": directories["node"],
        "NUMBA_CACHE_DIR": directories["numba"],
        "TORCH_HOME": directories["torch"],
        "JUPYTER_RUNTIME_DIR": directories["jupyter"],
    }
    for name, path in environment.items():
        os.environ[name] = str(path)

    # ``tempfile`` may have cached the Windows default before configuration.
    # Assigning this explicitly guarantees OpenSees recorder files use D:.
    tempfile.tempdir = str(directories["temp"])
    return {
        "project_root": str(project),
        "runtime_root": str(storage),
        "drive": storage.drive.upper(),
        "inside_project": True,
        "environment": {
            name: str(path) for name, path in environment.items()
        },
    }

