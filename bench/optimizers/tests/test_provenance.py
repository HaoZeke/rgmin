#!/usr/bin/env python3
"""Provenance tests for the optimizer coverage benchmark harness and artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

OPTIMIZERS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = OPTIMIZERS_DIR.parent.parent
if str(OPTIMIZERS_DIR) not in sys.path:
    sys.path.insert(0, str(OPTIMIZERS_DIR))

import run as bench_run  # noqa: E402

PUBLIC_REMOTES = {
    "rgmin": "https://github.com/OmniPotentRPC/rgmin.git",
    "rgpot": "https://github.com/OmniPotentRPC/rgpot.git",
    "rgpot_binding_patch": "https://github.com/OmniPotentRPC/rgpot.git",
    "eon": "https://github.com/TheochemUI/eOn.git",
    "eon_bundled_rgpot": "https://github.com/OmniPotentRPC/rgpot.git",
}

KNOWN_BACKENDS = (
    "openblas",
    "scipy-openblas",
    "mkl",
    "blis",
    "accelerate",
    "flexiblas",
    "netlib",
    "atlas",
)

HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
OFFLINE_MARKERS = (
    "could not resolve host",
    "failed to connect",
    "network is unreachable",
    "temporary failure in name resolution",
    "connection timed out",
    "connection refused",
    "ssl",
    "couldn't connect to server",
)


def validate_blas_string(value: Any, *, require_known_backend: bool = True) -> None:
    assert isinstance(value, str), f"blas value must be a string, got {type(value).__name__}"
    stripped = value.strip()
    assert stripped, "blas value must not be empty"
    assert not stripped.startswith("{") and not stripped.startswith("["), (
        f"blas value looks like a JSON fragment: {value!r}"
    )
    lowered = stripped.lower()
    if lowered.startswith("unknown"):
        assert re.match(r"^unknown\s*\(.+\)$", stripped), (
            f"unknown blas value must include an explicit reason: {value!r}"
        )
        if require_known_backend:
            pytest.fail(f"expected a detected BLAS backend, got {value!r}")
        return
    assert any(backend in lowered for backend in KNOWN_BACKENDS), (
        f"blas value does not name a recognized backend: {value!r}"
    )


def _is_offline_error(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in OFFLINE_MARKERS)


def _fetch_bare_repo(remote: str, dest: Path) -> None:
    proc = subprocess.run(
        ["git", "clone", "--filter=blob:none", "--bare", "--quiet", remote, str(dest)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        if _is_offline_error(stderr):
            pytest.skip(f"offline: cannot fetch public remote {remote}: {stderr}")
        pytest.fail(f"failed to fetch public remote {remote}: {stderr}")


def _assert_commit_reachable(git_dir: Path, source_key: str, sha: str, remote: str) -> None:
    assert HEX40_RE.match(sha), f"{source_key} is not a 40-char commit SHA: {sha!r}"
    cat = subprocess.run(
        ["git", f"--git-dir={git_dir}", "cat-file", "-t", sha],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cat.returncode == 0 and cat.stdout.strip() == "commit", (
        f"{source_key} SHA {sha} is not a commit object on {remote}: {cat.stderr.strip()}"
    )
    refs = subprocess.run(
        ["git", f"--git-dir={git_dir}", "for-each-ref", f"--contains={sha}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refs.returncode == 0 and refs.stdout.strip(), (
        f"{source_key} SHA {sha} is not reachable from any fetched public ref on {remote}"
    )


def test_blas_validator_rejects_json_fragments_and_accepts_backend_names() -> None:
    for bad in ("{", '{"blas": "openblas"}', "[", "[1, 2]", "", "unknown"):
        with pytest.raises(AssertionError):
            validate_blas_string(bad, require_known_backend=False)
    validate_blas_string("blas=scipy-openblas 0.3.34.237.2; runtime=openblas")
    validate_blas_string("unknown (numpy Build Dependencies missing)", require_known_backend=False)


def test_harness_blas_backend_reports_real_backend() -> None:
    detected = bench_run.blas_backend()
    validate_blas_string(detected, require_known_backend=True)


def test_harness_blas_backend_unknown_fallback_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench_run, "_numpy_blas_parts", lambda reasons: reasons.append("no numpy blas") or [])
    monkeypatch.setattr(bench_run, "_threadpoolctl_blas_parts", lambda reasons: reasons.append("no threadpoolctl") or [])
    fallback = bench_run.blas_backend()
    assert fallback.startswith("unknown (")
    validate_blas_string(fallback, require_known_backend=False)


def test_artifacts_blas_backend_is_valid() -> None:
    raw = json.loads((OPTIMIZERS_DIR / "results.json").read_text(encoding="utf-8"))
    analyzed = json.loads((OPTIMIZERS_DIR / "results-analyzed.json").read_text(encoding="utf-8"))
    validate_blas_string(raw["runtime"]["blas"], require_known_backend=True)
    validate_blas_string(analyzed["runtime"]["blas"], require_known_backend=True)
    report = (OPTIMIZERS_DIR / "results.md").read_text(encoding="utf-8")
    machine_lines = [line for line in report.splitlines() if line.startswith("Machine: ")]
    assert len(machine_lines) == 1
    assert "BLAS: {" not in machine_lines[0]
    assert "BLAS: [" not in machine_lines[0]


def test_recorded_sources_are_verifiable_on_public_remotes(tmp_path: Path) -> None:
    raw = json.loads((OPTIMIZERS_DIR / "results.json").read_text(encoding="utf-8"))
    sources = raw["sources"]

    patch_entry = sources.get("rgpot_binding_patch")
    if isinstance(patch_entry, dict):
        patch_path = REPO_ROOT / patch_entry["path"]
        assert patch_path.is_file(), f"recorded patch file missing: {patch_path}"
        assert OPTIMIZERS_DIR in patch_path.resolve().parents, (
            f"patch must live under {OPTIMIZERS_DIR}"
        )
        digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        assert digest == patch_entry["sha256"], "recorded patch sha256 mismatch"
        assert HEX40_RE.match(patch_entry["tree"]), "patched tree must be a 40-char SHA"

    needed_commits: list[tuple[str, str, str]] = []
    for key, remote in PUBLIC_REMOTES.items():
        value = sources.get(key)
        if value is None or isinstance(value, dict):
            continue
        needed_commits.append((key, value, remote))

    cloned: dict[str, Path] = {}
    for index, (key, sha, remote) in enumerate(needed_commits):
        if remote not in cloned:
            repo_dir = tmp_path / f"remote-{index}.git"
            _fetch_bare_repo(remote, repo_dir)
            cloned[remote] = repo_dir
        _assert_commit_reachable(cloned[remote], key, sha, remote)
