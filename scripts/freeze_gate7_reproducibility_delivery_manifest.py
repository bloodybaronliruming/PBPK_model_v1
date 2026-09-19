"""Freeze a no-label Gate 7 delivery manifest and clean-reload protocol.

The stage inventories already frozen assets, scripts, and the active oneadmet
environment.  It never loads a predictive model, reads labels, scores data, or
creates/modifies a Conda environment.  A later, separately confirmed clean
environment smoke is verified by ``verify_gate7_clean_environment_reload.py``.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


STAGE = "gate7_reproducibility_delivery_manifest"
PRECURSORS = {
    "r0": ("results/analysis/gate7_release_readiness_audit_v3", "gate7_release_readiness_audit"),
    "r1": ("data/public_development/gate7_release_contract_v2", "gate7_release_contract_freeze"),
    "r3": ("results/analysis/gate7_model_data_card_audit_v1", "gate7_model_data_card_availability_audit"),
    "r4": ("results/analysis/gate7_release_cli_smoke_v1", "gate7_dual_lane_batch_inference"),
    "r5": ("results/analysis/gate7_endpoint_evidence_matrix_v3", "gate7_endpoint_diagnostic_evidence_matrix"),
}
CRITICAL_SCRIPTS = [
    "infer_gate7_dual_lane.py",
    "smoke_gate7_release_interface.py",
    "gate5_clint_common.py",
    "infer_frozen_candidates.py",
    "pipeline_common.py",
    "verify_gate7_clean_environment_reload.py",
]
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
PYG_WHEEL_INDEX = "https://data.pyg.org/whl/torch-2.5.1+cu121.html"
PYTORCH_CUDA_PACKAGES = {"torch", "torchvision", "torchaudio"}
PYG_EXTENSION_PACKAGES = {"pyg-lib", "torch-cluster", "torch-scatter", "torch-sparse", "torch-spline-conv"}
CONDA_PIP_BOOTSTRAP_PACKAGES = {"packaging", "pip", "setuptools", "wheel"}


def checked_output(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def runtime_versions() -> dict:
    names = ["numpy", "pandas", "scikit-learn", "rdkit", "joblib", "torch", "matplotlib"]
    versions = {name: importlib.metadata.version(name) for name in names}
    try:
        import torch
        torch_info = {
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_build": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # Diagnostic only; a missing optional runtime must be explicit.
        torch_info = {"torch_probe_error": f"{type(exc).__name__}: {exc}"}
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "torch": torch_info,
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def requirement_name(requirement: str) -> str:
    """Return a normalized name for a pinned simple requirement."""
    return requirement.split(" @ ", 1)[0].split("==", 1)[0].lower().replace("_", "-")


def split_environment_requirements(environment_export: str, pip_freeze: str) -> tuple[str, list[str], list[str], list[str]]:
    """Keep non-PyPI CUDA wheels out of the generic pip transaction.

    PyTorch CUDA and PyG extension wheels have deliberately pinned local version
    tags (for example ``+cu121``). They are unavailable from the ordinary PyPI
    index, so they must be installed from their provenance-specific indexes
    before the remaining ordinary requirements.
    """
    marker = "\n  - pip:\n"
    if marker not in environment_export:
        raise ValueError("Full Conda export has no pip section to separate")
    conda_environment = environment_export.split(marker, 1)[0].rstrip() + "\n"
    pytorch, pyg, standard = [], [], []
    for raw in pip_freeze.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = requirement_name(line)
        if name in CONDA_PIP_BOOTSTRAP_PACKAGES:
            continue
        if name in PYTORCH_CUDA_PACKAGES:
            pytorch.append(line)
        elif name in PYG_EXTENSION_PACKAGES:
            pyg.append(line)
        else:
            standard.append(line)
    if len(pytorch) != 3 or len(pyg) != 5:
        raise ValueError(f"Expected 3 PyTorch CUDA and 5 PyG extension requirements, found {len(pytorch)} and {len(pyg)}")
    return conda_environment, pytorch, pyg, standard


def asset_manifest(root: Path) -> tuple[dict, dict[str, Path]]:
    folders = {}
    for key, (relative, stage) in PRECURSORS.items():
        folder = root / relative
        verify_stage(folder, stage)
        folders[key] = folder
    r4 = read_json(folders["r4"] / "complete.json")
    if not (r4.get("no_labels_accessed") and r4.get("technical_smoke") and r4.get("output_rows") == 14):
        raise ValueError("Gate 7 R4 does not provide the expected label-free reference smoke")
    smoke_input = root / "data/public_development/gate7_release_cli_smoke_input_v1.csv"
    reference_prediction = folders["r4"] / "dual_lane_predictions.csv"
    if sha256(smoke_input) != sha256(folders["r4"] / "submitted_label_free_input.csv"):
        raise ValueError("R4 submitted smoke input differs from the frozen R6 replay input")
    sources = {
        "AGENTS.md": root / "AGENTS.md",
        "gate7_cli_smoke_input": smoke_input,
        "r4_reference_predictions": reference_prediction,
        "r1_release_contract": folders["r1"] / "release_contract.json",
        "r3_endpoint_cards": folders["r3"] / "endpoint_cards.json",
        "r0_readiness_table": folders["r0"] / "release_readiness_by_endpoint.csv",
    }
    sources.update({f"script::{name}": root / "scripts" / name for name in CRITICAL_SCRIPTS})
    if any(not path.is_file() for path in sources.values()):
        missing = [str(path) for path in sources.values() if not path.is_file()]
        raise FileNotFoundError(f"Delivery manifest inputs missing: {missing}")
    manifest = {
        "manifest_version": 6,
        "scope": "Gate 7 technical dual-lane inference only; not a unified numeric release",
        "active_environment": "oneadmet",
        "model_or_label_accessed": False,
        "predecessors": {
            key: {
                "path": str(folder.resolve()),
                "complete_sha256": sha256(folder / "complete.json"),
            }
            for key, folder in folders.items()
        },
        "critical_file_sha256": {key: sha256(path) for key, path in sources.items()},
        "reference_replay": {
            "input": str(smoke_input.resolve()),
            "input_sha256": sha256(smoke_input),
            "reference_r4_output": str(reference_prediction.resolve()),
            "reference_prediction_sha256": sha256(reference_prediction),
            "required_command_confirmation": "--confirm-no-label-inference",
            "required_technical_smoke_flag": "--technical-smoke",
            "expected_input_rows": 2,
            "expected_output_rows": 14,
            "cross_process_prediction_tolerance": 1e-12,
            "in_process_repeat_tolerance": 1e-12,
            "reference_in_process_repeat_difference": float(r4["max_abs_repeat_prediction_difference"]),
            "numeric_release_authorized": False,
            "f_rule": "status-only; predicted_physical missing",
        },
        "environment_restore_warning": (
            "The active oneadmet environment includes pip-installed packages. Conda builds, PyTorch CUDA wheels, PyG extension "
            "wheels and ordinary pip packages are restored in separate provenance-specific transactions. The explicit and pip-freeze "
            "snapshots are retained as independent audit records; Conda history alone is intentionally insufficient."
        ),
    }
    return manifest, folders


def protocol_text() -> str:
    return """# Gate 7 independent clean-environment reload protocol

## Status and authorization

This protocol is frozen by R6 but is **not executed** by R6. It creates a new
environment and may download packages, so it requires explicit user approval.
It must never modify `oneadmet`, overwrite an existing output directory, read
labels, calculate performance, calibrate a model, or replace a candidate.

## Inputs frozen by this manifest

- `conda_environment.yml` pins the complete Conda package builds. Pip packages
  are split by provenance: `pytorch_cuda_requirements.txt` is installed from the
  PyTorch CUDA 12.1 index; `pyg_extension_requirements.txt` is installed from
  the matching PyG wheel page; `pip_requirements_standard.txt` uses the ordinary
  configured Pip index. `conda_explicit.txt` and `pip_freeze.txt` are retained
  as independent audit snapshots.
  `conda_history.yml` is documentation only and is not sufficient for restore.
- Use the contract-bound R4 input `gate7_release_cli_smoke_input_v1.csv`.
- The reference hash is for R4's **label-free technical predictions**, not for
  any test score or model-selection output.

## Approved execution sequence after user authorization

Run from the repository root, with a new environment name and new output paths:

```bash
MANIFEST=data/public_development/gate7_reproducibility_delivery_manifest_v8
RELOAD_ENV=oneadmet-gate7-reload-v4
RELOAD_SMOKE=results/analysis/gate7_clean_environment_reload_smoke_v4
RELOAD_VERIFY=results/analysis/gate7_clean_environment_reload_verification_v4

conda env create --yes --name "$RELOAD_ENV" --file "$MANIFEST/conda_environment.yml"
conda run --no-capture-output -n "$RELOAD_ENV" python -m pip install \
  --index-url "https://download.pytorch.org/whl/cu121" \
  --extra-index-url "https://pypi.tuna.tsinghua.edu.cn/simple" \
  --requirement "$MANIFEST/pytorch_cuda_requirements.txt"
conda run --no-capture-output -n "$RELOAD_ENV" python -m pip install \
  --find-links "https://data.pyg.org/whl/torch-2.5.1+cu121.html" \
  --requirement "$MANIFEST/pyg_extension_requirements.txt"
conda run --no-capture-output -n "$RELOAD_ENV" python -m pip install \
  --requirement "$MANIFEST/pip_requirements_standard.txt"
conda run --no-capture-output -n "$RELOAD_ENV" python -m pip check
conda run --no-capture-output -n "$RELOAD_ENV" python scripts/infer_gate7_dual_lane.py \
  --input data/public_development/gate7_release_cli_smoke_input_v1.csv \
  --output "$RELOAD_SMOKE" --technical-smoke --confirm-no-label-inference
conda run --no-capture-output -n "$RELOAD_ENV" python scripts/verify_gate7_clean_environment_reload.py \
  --manifest "$MANIFEST" --candidate-smoke "$RELOAD_SMOKE" \
  --output "$RELOAD_VERIFY" --confirm-clean-reload-verification
```

## Passing conditions

1. Both new output directories contain valid, non-partial `complete.json`.
2. Smoke has 2 inputs, 14 rows, 12 finite numeric technical outputs, two F
   status-only rows and an in-process repeat difference no greater than `1e-12`.
3. Candidate contract/card/input hashes equal the R6 manifest.
4. The submitted input SHA-256 equals the frozen R4 technical reference. The
   prediction table has an identical schema and non-prediction fields, while
   numeric technical outputs differ by no more than `1e-12` in absolute value.
   The CSV byte SHA-256 is retained as an audit record but is not a cross-process
   pass/fail rule, because last-place float serialization can differ.
5. The verification report records no label access, scoring, fitting,
   calibration or candidate change.

If package solve/install fails, a hash differs, model loading fails, output is
non-finite, F becomes numeric, or the output directory already exists: stop,
preserve the failed log/output, and report it. Do not repair the frozen model or
repeat any test evaluation.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "data/public_development/gate7_reproducibility_delivery_manifest_v8"
    manifest, folders = asset_manifest(root)
    conda_export = checked_output(["conda", "env", "export", "--name", "oneadmet"])
    pip_freeze = checked_output([sys.executable, "-m", "pip", "freeze", "--all"])
    conda_environment, pytorch_requirements, pyg_requirements, standard_requirements = split_environment_requirements(conda_export, pip_freeze)
    if args.check_only:
        print(f"Gate 7 R6 preflight: predecessors={len(folders)}; no models, labels, or environment mutation.")
        return
    with stage_output(output) as folder:
        (folder / "delivery_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "conda_history.yml").write_text(checked_output(["conda", "env", "export", "--name", "oneadmet", "--from-history"]), encoding="utf-8")
        (folder / "conda_environment.yml").write_text(conda_environment, encoding="utf-8")
        (folder / "conda_explicit.txt").write_text(checked_output(["conda", "list", "--name", "oneadmet", "--explicit"]), encoding="utf-8")
        (folder / "pip_freeze.txt").write_text(pip_freeze, encoding="utf-8")
        (folder / "pytorch_cuda_requirements.txt").write_text("\n".join(pytorch_requirements) + "\n", encoding="utf-8")
        (folder / "pyg_extension_requirements.txt").write_text("\n".join(pyg_requirements) + "\n", encoding="utf-8")
        (folder / "pip_requirements_standard.txt").write_text("\n".join(standard_requirements) + "\n", encoding="utf-8")
        (folder / "runtime_versions.json").write_text(json.dumps(runtime_versions(), ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "clean_environment_reload_protocol.md").write_text(protocol_text(), encoding="utf-8")
        (folder / "README.md").write_text(
            "# Gate 7 R6 reproducibility delivery manifest\n\n"
            "This is an inventory and protocol only. It captures the current oneadmet Conda/Pip runtime, critical code/data "
            "and technical-reference hashes without loading models or labels. Nonstandard CUDA/PyG wheel sources are explicit. "
            "It does not create an environment. The separate clean-environment replay remains user-authorized work.\n",
            encoding="utf-8",
        )
        inputs = {str((folder_path / "complete.json").resolve()): sha256(folder_path / "complete.json") for folder_path in folders.values()}
        finish_stage(
            folder, STAGE, inputs=inputs, partial=False, no_model_loaded=True, no_labels_accessed=True,
            no_prediction_values_interpreted=True, no_performance_metrics_computed=True, no_model_fitted=True,
            no_calibration=True, no_candidate_selection_changed=True, environment_created=False,
            environment_packages_installed=False, clean_reload_executed=False, critical_file_count=len(manifest["critical_file_sha256"]),
        )
    print(f"Gate 7 R6 reproducibility delivery manifest: {output}")


if __name__ == "__main__":
    run_cli(main)
