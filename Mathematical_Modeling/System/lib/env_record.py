"""实验环境记录工具：固定随机种子、记录版本、计算可复现指纹。

用法：在实验脚本开头
    from env_record import snapshot, seed_all
    seed_all(42)
    ...
    snapshot(out_dir, seed=42, data_files=[...], config={...})
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

# 关键库版本（缺失记为 None，不估算）
_PKGS = [
    "numpy", "scipy", "pandas", "scikit-learn", "matplotlib",
    "networkx", "statsmodels", "ortools", "highspy", "numba",
]


def seed_all(seed: int = 42) -> None:
    """固定所有能固定的随机源。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _pkg_versions() -> dict:
    out = {}
    for name in _PKGS:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = None
    return out


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def snapshot(out_dir: str | Path, *, seed: int, data_files=None,
             config=None, extra=None) -> dict:
    """写出 env.json 并返回 dict。不伪造任何拿不到的字段。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_files = data_files or []
    repo = Path(__file__).resolve().parents[3]  # Note/
    info = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": _pkg_versions(),
        "seed": seed,
        "git_commit": _git_commit(repo),
        "config": config or {},
        "data": [{"path": str(p), "sha256": _sha256(p)} for p in data_files],
    }
    if extra:
        info["extra"] = extra
    (out_dir / "env.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def result_fingerprint(obj) -> str:
    """对结果 dict 做稳定哈希：用于复跑一致性校验（忽略耗时字段）。"""
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items()
                    if not any(t in k.lower() for t in ("time", "runtime", "elapsed", "seed_used",
                                                       "captured_at", "fingerprint", "ts", "timing"))}
        if isinstance(o, list):
            return [strip(v) for v in o]
        if isinstance(o, float):
            return round(o, 10)
        return o
    return hashlib.sha256(
        json.dumps(strip(obj), ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]
