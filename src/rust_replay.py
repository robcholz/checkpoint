from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_MODULE_NAME = "_gockpt_rust_replay"
_MODULE: ModuleType | None = None


def _candidate_paths() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    release_dir = repo_root / "rust" / "gockpt_rust_replay" / "target" / "release"
    return [
        release_dir / "lib_gockpt_rust_replay.so",
        release_dir / "_gockpt_rust_replay.so",
    ]


def load_rust_replay() -> ModuleType | None:
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    for path in _candidate_paths():
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
        _MODULE = module
        return module
    return None


def adamw_update(*args: Any, **kwargs: Any) -> int | None:
    module = load_rust_replay()
    if module is None:
        return None
    return module.adamw_update(*args, **kwargs)
