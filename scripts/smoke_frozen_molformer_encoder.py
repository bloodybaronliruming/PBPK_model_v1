#!/usr/bin/env python3
"""Run a label-free, revision-pinned MoLFormer embedding smoke.

The smoke audits the locally cached remote code before loading it, tokenizes all
frozen parents without truncation, and embeds a deterministic 32-parent sample.
It checks repeatability, CPU/GPU agreement, non-finite outputs, reload integrity,
and writes no split labels or endpoint values.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"
MAX_TOKENS = 202
HIDDEN_SIZE = 768
REMOTE_CODE_FILES = [
    "configuration_molformer.py",
    "modeling_molformer.py",
    "tokenization_molformer.py",
    "tokenization_molformer_fast.py",
]


def default_snapshot() -> Path:
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    return cache / "hub/models--ibm-research--MoLFormer-XL-both-10pct/snapshots" / REVISION


def audit_remote_code(snapshot: Path) -> pd.DataFrame:
    forbidden = {
        "subprocess_import": re.compile(r"(^|\n)\s*(from\s+subprocess\s+import|import\s+subprocess)"),
        "network_import": re.compile(r"(^|\n)\s*(from\s+(requests|urllib|socket)\s+import|import\s+(requests|urllib|socket))"),
        "shell_execution": re.compile(r"\b(os\.system|os\.popen|subprocess\.)"),
        "dynamic_execution": re.compile(r"\b(eval|exec)\s*\("),
        "destructive_fs": re.compile(r"\b(shutil\.rmtree|os\.remove|os\.unlink)\s*\("),
        "unsafe_deserialization": re.compile(r"\b(pickle\.loads|torch\.load)\s*\("),
    }
    rows = []
    for name in REMOTE_CODE_FILES:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(f"Pinned remote-code file missing: {path}")
        text = path.read_text(encoding="utf-8")
        hits = [label for label, pattern in forbidden.items() if pattern.search(text)]
        rows.append({
            "file": name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "forbidden_pattern_hits": "|".join(hits),
            "audit_pass": not hits,
        })
    frame = pd.DataFrame(rows)
    if not frame.audit_pass.all():
        raise ValueError("Pinned executed remote code failed static audit")
    return frame


def token_inventory(tokenizer, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    unk_id = tokenizer.unk_token_id
    for row in manifest.itertuples(index=False):
        ids = tokenizer.encode(row.graph_smiles, add_special_tokens=True, truncation=False)
        rows.append({
            "molecule_id": row.molecule_id,
            "feature_index": int(row.feature_index),
            "graph_smiles": row.graph_smiles,
            "smiles_sha256": __import__("hashlib").sha256(row.graph_smiles.encode("utf-8")).hexdigest(),
            "token_count": len(ids),
            "unknown_token_count": int(sum(token == unk_id for token in ids)) if unk_id is not None else 0,
            "within_model_limit": len(ids) <= MAX_TOKENS,
            "failure_reason": "" if len(ids) <= MAX_TOKENS else "token_overflow_no_truncation",
        })
    return pd.DataFrame(rows)


def select_smoke_rows(tokens: pd.DataFrame, n: int) -> pd.DataFrame:
    valid = tokens.loc[tokens.within_model_limit].sort_values(
        ["token_count", "molecule_id"], kind="stable"
    ).reset_index(drop=True)
    if len(valid) < n:
        raise ValueError(f"Only {len(valid)} parents fit the model token limit; need {n}")
    indices = np.rint(np.linspace(0, len(valid) - 1, n)).astype(int)
    selected = valid.iloc[indices].copy().reset_index(drop=True)
    if selected.molecule_id.duplicated().any():
        raise ValueError("Quantile smoke selection produced duplicate parents")
    selected.insert(0, "smoke_index", np.arange(len(selected)))
    return selected


def embed(model, tokenizer, smiles: list[str], device: torch.device, batch_size: int) -> np.ndarray:
    model.to(device)
    model.eval()
    arrays = []
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            batch = tokenizer(
                smiles[start:start + batch_size], padding=True, truncation=False, return_tensors="pt"
            )
            if int(batch["attention_mask"].sum(dim=1).max()) > MAX_TOKENS:
                raise ValueError("Smoke embedding received an overflowing sequence")
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch).pooler_output
            arrays.append(output.detach().float().cpu().numpy())
    result = np.concatenate(arrays, axis=0)
    if result.shape != (len(smiles), HIDDEN_SIZE) or not np.isfinite(result).all():
        raise ValueError(f"Invalid embedding shape or values: {result.shape}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-protocol", type=Path, default=ROOT / "data/public_development/algorithm_landscape_smiles_provider_v1")
    parser.add_argument("--graph-manifest", type=Path, default=ROOT / "data/public_development/gate1b_graph_manifest_v1")
    parser.add_argument("--snapshot", type=Path, default=default_snapshot())
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/molformer_embedding_smoke_v1")
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    required = [
        args.provider_protocol / "complete.json", args.provider_protocol / "smiles_provider_registry.csv",
        args.graph_manifest / "complete.json", args.graph_manifest / "graph_manifest.csv",
        args.snapshot / "config.json", args.snapshot / "model.safetensors",
        args.snapshot / "tokenizer.json", *[args.snapshot / name for name in REMOTE_CODE_FILES],
    ]
    startup_self_check(required, output=args.output)
    protocol = verify_stage(args.provider_protocol, "algorithm_landscape_smiles_provider_protocol")
    graph = verify_stage(args.graph_manifest, "gate1b_graph_manifest")
    if protocol.get("primary_revision") != REVISION or int(graph.get("parents", 0)) != 7939:
        raise ValueError("Provider revision or frozen parent population changed")
    if protocol.get("validation_labels_read") or protocol.get("test_labels_read") or graph.get("test_labels_read"):
        raise ValueError("Upstream stage reports protected-label access")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Gate1B encoder smoke")

    code_audit = audit_remote_code(args.snapshot)
    providers = pd.read_csv(args.provider_protocol / "smiles_provider_registry.csv")
    primary = providers.loc[providers.provider_role.eq("primary")]
    if len(primary) != 1 or primary.iloc[0].immutable_revision != REVISION:
        raise ValueError("Primary provider registry does not match the smoke revision")

    # Loading from a local immutable snapshot prevents runtime revision drift or network fallback.
    tokenizer = AutoTokenizer.from_pretrained(
        args.snapshot, trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        args.snapshot, trust_remote_code=True, local_files_only=True, deterministic_eval=True
    )
    if int(model.config.max_position_embeddings) != MAX_TOKENS or int(model.config.hidden_size) != HIDDEN_SIZE:
        raise ValueError("Loaded model dimensions do not match the frozen provider contract")
    if not bool(model.config.deterministic_eval):
        raise ValueError("MoLFormer deterministic_eval was not enabled")

    manifest = pd.read_csv(args.graph_manifest / "graph_manifest.csv", dtype={"molecule_id": str})
    if len(manifest) != 7939 or manifest.molecule_id.duplicated().any() or manifest.graph_smiles.isna().any():
        raise ValueError("Graph manifest is incomplete or not parent-keyed")
    inventory = token_inventory(tokenizer, manifest)
    selected = select_smoke_rows(inventory, args.sample_size)
    smiles = selected.graph_smiles.tolist()

    gpu_device = torch.device("cuda:0")
    gpu_first = embed(model, tokenizer, smiles, gpu_device, args.batch_size)
    gpu_repeat = embed(model, tokenizer, smiles, gpu_device, args.batch_size)
    repeat_max_abs = float(np.max(np.abs(gpu_first - gpu_repeat)))
    if repeat_max_abs > 1e-6:
        raise ValueError(f"Same-device repeatability failed: {repeat_max_abs:.3g}")

    cpu_indices = np.rint(np.linspace(0, len(smiles) - 1, min(8, len(smiles)))).astype(int)
    cpu_smiles = [smiles[index] for index in cpu_indices]
    cpu_embedding = embed(model, tokenizer, cpu_smiles, torch.device("cpu"), batch_size=4)
    gpu_subset = embed(model, tokenizer, cpu_smiles, gpu_device, batch_size=4)
    cpu_gpu_max_abs = float(np.max(np.abs(cpu_embedding - gpu_subset)))
    if cpu_gpu_max_abs > 1e-4:
        raise ValueError(f"CPU/GPU consistency failed: {cpu_gpu_max_abs:.3g}")

    input_hashes = {
        "provider_protocol_complete_sha256": sha256(args.provider_protocol / "complete.json"),
        "graph_manifest_complete_sha256": sha256(args.graph_manifest / "complete.json"),
        "model_safetensors_sha256": sha256(args.snapshot / "model.safetensors"),
        "tokenizer_json_sha256": sha256(args.snapshot / "tokenizer.json"),
    }
    summary = {
        "model_id": MODEL_ID,
        "immutable_revision": REVISION,
        "parents_tokenized": len(inventory),
        "parents_within_limit": int(inventory.within_model_limit.sum()),
        "parents_overflow": int((~inventory.within_model_limit).sum()),
        "parents_with_unknown_tokens": int(inventory.unknown_token_count.gt(0).sum()),
        "maximum_observed_tokens": int(inventory.token_count.max()),
        "smoke_parents": len(selected),
        "embedding_dimension": HIDDEN_SIZE,
        "gpu_repeat_max_abs_difference": repeat_max_abs,
        "cpu_gpu_max_abs_difference": cpu_gpu_max_abs,
        "remote_code_audit_passed": bool(code_audit.audit_pass.all()),
        "cuda_device": torch.cuda.get_device_name(0),
        "validation_labels_read": False,
        "test_labels_read": False,
        "model_selection_authorized": False,
        "full_cache_authorized": True,
    }
    with stage_output(args.output) as out:
        selected.drop(columns=["graph_smiles"]).to_csv(out / "smoke_parent_manifest.csv", index=False)
        inventory.drop(columns=["graph_smiles"]).to_csv(out / "token_coverage_inventory.csv", index=False)
        code_audit.to_csv(out / "remote_code_static_audit.csv", index=False)
        np.save(out / "smoke_embeddings_float32.npy", gpu_first.astype(np.float32, copy=False))
        pd.DataFrame({
            "smoke_index": cpu_indices,
            "molecule_id": selected.iloc[cpu_indices].molecule_id.to_numpy(),
            "cpu_gpu_row_max_abs_difference": np.max(np.abs(cpu_embedding - gpu_subset), axis=1),
        }).to_csv(out / "cpu_gpu_consistency.csv", index=False)
        saved = np.load(out / "smoke_embeddings_float32.npy")
        if saved.shape != gpu_first.shape or not np.array_equal(saved, gpu_first.astype(np.float32)):
            raise ValueError("Saved smoke embeddings failed exact reload integrity")
        (out / "runtime_environment.json").write_text(json.dumps({
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(0),
        }, indent=2) + "\n", encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Frozen MoLFormer encoder smoke v1\n\n"
            "Label-free smoke at an immutable model revision. The full parent population was tokenized without truncation; a deterministic length-stratified 32-parent sample was embedded. "
            "Remote executed code, GPU repeatability, CPU/GPU agreement, finite outputs and exact cache reload were checked. This artifact cannot select a model or open validation/test labels.\n",
            encoding="utf-8",
        )
        finish_stage(out, "frozen_molformer_encoder_smoke", inputs=input_hashes, partial=False, **summary)
    print(f"Frozen MoLFormer smoke: {args.output}")


if __name__ == "__main__":
    run_cli(main)
