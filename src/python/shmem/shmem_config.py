#!/usr/bin/env python
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import argparse
import json
import os
from pathlib import Path

from ._soc import detect_soc, select_backend, _read_proc_maps

_PKG_ROOT = Path(__file__).resolve().parent


def _read_version():
    version_file = _PKG_ROOT / "VERSION"
    if version_file.exists():
        first_line = version_file.read_text(encoding="utf-8").splitlines()[0].strip()
        return first_line if first_line else "unknown"
    return "unknown"


def _get_include_dir():
    return str(_PKG_ROOT / "include")


def _get_lib_dir():
    backend = select_backend()
    backend_dir = _PKG_ROOT / "backends" / backend
    if backend_dir.exists():
        return str(backend_dir)
    return str(_PKG_ROOT)


def _get_root_dir():
    return str(_PKG_ROOT)


def _get_backend():
    return select_backend()


def _get_ldflags():
    include_dir = _get_include_dir()
    lib_dir = _get_lib_dir()
    return f"-I{include_dir} -L{lib_dir} -lshmem"


def _get_rpath():
    lib_dir = _get_lib_dir()
    return f"-Wl,-rpath,{lib_dir}"


def _get_runtime_root():
    for line in _read_proc_maps():
        if "libshmem.so" in line:
            parts = line.split()
            if len(parts) >= 6:
                lib_path = Path(parts[5]).resolve()
                return str(lib_path.parents[2])  # .../backends/910/libshmem.so → .../shmem/ (.parent × 3)

    libshmem_path = os.environ.get("LD_LIBRARY_PATH", "")
    for p in libshmem_path.split(":"):
        candidate = Path(p) / "libshmem.so"
        if candidate.exists():
            return str(candidate.resolve().parents[2])  # .../backends/910/libshmem.so → .../shmem/ (.parent × 3)

    return ""


def _detect_multi_so_conflict():
    loaded_paths = []

    for line in _read_proc_maps():
        if "libshmem.so" in line:
            parts = line.split()
            if len(parts) >= 6:
                loaded_paths.append(parts[5])

    seen = set()
    unique_paths = []
    for p in loaded_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    conflict = len(unique_paths) > 1
    return conflict, unique_paths



_BACKEND_REQUIRED_SO = [
    "libshmem.so",
    "libshmem_utils.so",
    "aclshmem_bootstrap_config_store.so",
    "aclshmem_bootstrap_uid.so",
]


def _check_backend_artifacts():
    backend = select_backend()
    backend_dir = _PKG_ROOT / "backends" / backend
    missing = []

    if not backend_dir.exists():
        missing.append(f"backends/{backend}/ directory not found")
    else:
        for so_name in _BACKEND_REQUIRED_SO:
            so_file = backend_dir / so_name
            if not so_file.exists():
                missing.append(f"backends/{backend}/{so_name} not found")

    # pybind11 binding
    pyshmem_files = list(_PKG_ROOT.glob("_pyshmem*.so"))
    if not pyshmem_files:
        missing.append("_pyshmem.so not found")

    ok = len(missing) == 0
    return ok, missing, backend


def _is_release_build():
    """Return True for release builds (no build_type: debug in VERSION)."""
    version_file = _PKG_ROOT / "VERSION"
    if version_file.exists():
        return "build_type: debug" not in version_file.read_text(encoding="utf-8")
    return True


def cmd_include():
    print(_get_include_dir())


def cmd_lib():
    print(_get_lib_dir())


def cmd_backend():
    print(select_backend())


def cmd_root():
    print(_get_root_dir())


def cmd_version():
    print(_read_version())


def cmd_runtime_root():
    print(_get_runtime_root())


def cmd_ldflags():
    print(_get_ldflags())


def cmd_rpath():
    print(_get_rpath())


def cmd_diagnose():
    result = {}
    result["version"] = _read_version()

    backend = select_backend()
    detected_soc = detect_soc()
    result["backend"] = {
        "selected": backend,
        "auto_detected_soc": detected_soc or "unknown",
    }

    result["release_build"] = _is_release_build()

    conflict, conflict_paths = _detect_multi_so_conflict()
    result["multi_so_conflict"] = {
        "detected": conflict,
        "loaded_paths": conflict_paths,
    }

    backend_ok, backend_missing, backend_name = _check_backend_artifacts()
    result["backend_artifacts"] = {
        "backend": backend_name,
        "complete": backend_ok,
        "missing": backend_missing,
    }

    runtime_root = _get_runtime_root()
    pkg_root = _get_root_dir()
    result["runtime_root"] = {
        "path": runtime_root or "unknown",
        "matches_package_root": (runtime_root == pkg_root) if runtime_root else False,
        "package_root": pkg_root,
    }

    has_errors = conflict or not backend_ok
    if detected_soc is None:
        result["degraded"] = True
        result["degraded_reason"] = "Auto SoC detection failed, fallback to default 910 backend"
    else:
        result["degraded"] = False

    next_steps = []
    if conflict:
        next_steps.append(
            "Multiple libshmem.so paths detected in process. "
            "Remove duplicate installations or unset conflicting LD_LIBRARY_PATH entries."
        )
    if not backend_ok:
        next_steps.append(
            f"Artifacts missing for {backend_name}: {backend_missing}. "
            "Rebuild wheel with correct SOC_TYPE and verify all .so files are included."
        )
    if result["degraded"] and detected_soc is None:
        next_steps.append(
            "Auto SoC detection failed. Ensure Ascend driver is loaded and devices are visible."
        )
    if not has_errors and not result.get("degraded"):
        next_steps.append("No issues detected. SHMEM is ready.")

    result["next_steps"] = next_steps

    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="shmem-config",
        description="SHMEM configuration utility - query paths, backend, and diagnostics",
    )
    parser.add_argument("--include", action="store_true", help="Output C/C++ header include path")
    parser.add_argument("--lib", action="store_true", help="Output current backend library path")
    parser.add_argument("--backend", action="store_true", help="Output backend selection result")
    parser.add_argument("--root", action="store_true", help="Output package root path")
    parser.add_argument("--version", action="store_true", help="Output wheel VERSION")
    parser.add_argument("--runtime-root", action="store_true", help="Output runtime libshmem.so root path")
    parser.add_argument("--ldflags", action="store_true", help="Output recommended link flags (include/lib/rpath)")
    parser.add_argument("--rpath", action="store_true", help="Output recommended rpath argument")
    parser.add_argument("--diagnose", action="store_true", help="Output structured diagnostic information (JSON)")

    args = parser.parse_args()

    if args.include:
        cmd_include()
    elif args.lib:
        cmd_lib()
    elif args.backend:
        cmd_backend()
    elif args.root:
        cmd_root()
    elif args.version:
        cmd_version()
    elif args.runtime_root:
        cmd_runtime_root()
    elif args.ldflags:
        cmd_ldflags()
    elif args.rpath:
        cmd_rpath()
    elif args.diagnose:
        cmd_diagnose()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
