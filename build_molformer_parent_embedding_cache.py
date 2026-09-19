#!/usr/bin/env python3
"""Materialize the immutable, label-free MoLFormer parent embedding cache."""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage
from smoke_frozen_molformer_encoder import (
    HIDDEN_SIZE, MAX_TOKENS, MODEL_ID, REMOTE_CODE_FILES, REVISION, audit_remote_code, default_snapshot,
)


def embed_all(model, tokenizer, frame: pd.DataFrame, device: torch.device, batch_size: int) -> np.ndarray:
    result = np.empty((len(frame), HIDDEN_SIZE), dtype=np.float32)
    model.to(device).eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(frame), batch_size), desc="MoLFormer parent embeddings"):
            stop = min(start + batch_size, len(frame))
            smiles = frame.graph_smiles.iloc[start:stop].tolist()
            batch = tokenizer(smiles, padding=True, truncation=False, return_tensors="pt")
            lengths = batch["attention_mask"].sum(dim=1)
            if int(lengths.max()) > MAX_TOKENS:
                raise ValueError("Full cache received token overflow despite passed smoke inventory")
            batch = {key: value.to(device) for key, value in batch.items()}
            values = model(**batch).pooler_output.detach().float().cpu().numpy()
            if values.shape != (stop - start, HIDDEN_SIZE) or not np.isfinite(values).all():
                raise ValueError(f"Invalid embedding batch at rows {start}:{stop}")
            result[start:stop] = values
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-protocol", type=Path, default=ROOT / "data/public_development/algorithm_landscape_smiles_provider_v1")
    parser.add_argument("--graph-manifest", type=Path, default=ROOT / "data/public_development/gate1b_graph_manifest_v1")
    parser.add_argument("--smoke", type=Path, default=ROOT / "results/benchmarks/molformer_embedding_smoke_v1")
    parser.add_argument("--snapshot", type=Path, default=default_snapshot())
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/molformer_parent_embedding_cache_v1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [
        args.provider_protocol / "complete.json", args.provider_protocol / "embedding_cache_contract.csv",
        args.graph_manifest / "complete.json", args.graph_manifest / "graph_manifest.csv",
        args.smoke / "complete.json", args.smoke / "token_coverage_inventory.csv",
        args.snapshot / "config.json", args.snapshot / "model.safetensors", args.snapshot / "tokenizer.json",
        *[args.snapshot / name for name in REMOTE_CODE_FILES],
    ]
    startup_self_check(required, output=None if args.check_only else args.output)
    protocol = verify_stage(args.provider_protocol, "algorithm_landscape_smiles_provider_protocol")
    graph = verify_stage(args.graph_manifest, "gate1b_graph_manifest")
    smoke = verify_stage(args.smoke, "frozen_molformer_encoder_smoke")
    if protocol.get("primary_revision") != REVISION or smoke.get("immutable_revision") != REVISION:
        raise ValueError("Provider/smoke revision mismatch")
    if not smoke.get("full_cache_authorized") or int(smoke.get("parents_within_limit", 0)) != 7939:
        raise ValueError("The frozen encoder smoke did not authorize full-cache materialization")
    for meta in (protocol, graph, smoke):
        if meta.get("validation_labels_read") or meta.get("test_labels_read"):
            raise ValueError("An upstream stage reports protected-label access")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.check_only:
        print(f"MoLFormer full cache ready: parents={smoke['parents_within_limit']} batch_size={args.batch_size} device_required=cuda")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full MoLFormer cache materialization")

    code_audit = audit_remote_code(args.snapshot)
    manifest = pd.read_csv(args.graph_manifest / "graph_manifest.csv", dtype={"molecule_id": str})
    coverage = pd.read_csv(args.smoke / "token_coverage_inventory.csv", dtype={"molecule_id": str})
    if len(manifest) != 7939 or len(coverage) != 7939:
        raise ValueError("Expected 7,939 frozen parents in manifest and token inventory")
    frame = manifest.merge(
        coverage[["molecule_id", "smiles_sha256", "token_count", "unknown_token_count", "within_model_limit"]],
        on="molecule_id", how="left", validate="one_to_one",
    ).sort_values("molecule_id", kind="stable").reset_index(drop=True)
    calculated = frame.graph_smiles.map(lambda value: __import__("hashlib").sha256(value.encode("utf-8")).hexdigest())
    if frame.isna().any().any() or not calculated.equals(frame.smiles_sha256):
        raise ValueError("Graph manifest and smoke token inventory no longer have identical SMILES")
    if not frame.within_model_limit.all() or frame.unknown_token_count.ne(0).any():
        raise ValueError("Full cache refuses overflow or unknown-token parents")

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.snapshot, trust_remote_code=True, local_files_only=True, deterministic_eval=True
    )
    if int(model.config.max_position_embeddings) != MAX_TOKENS or int(model.config.hidden_size) != HIDDEN_SIZE:
        raise ValueError("Loaded model dimensions differ from the frozen contract")
    device = torch.device("cuda:0")
    matrix = embed_all(model, tokenizer, frame, device, args.batch_size)

    # Repeat the exact first production batch.  Repacking isolated rows changes
    # padding and therefore the floating-point reduction path; that is a useful
    # batch-shape sensitivity diagnostic, but not the predeclared same-batch
    # determinism check.
    audit_indices = np.arange(min(args.batch_size, len(frame)))
    repeat = embed_all(
        model, tokenizer, frame.iloc[audit_indices].reset_index(drop=True), device, args.batch_size
    )
    repeat_difference = np.max(np.abs(matrix[audit_indices] - repeat), axis=1)
    if float(repeat_difference.max()) > 1e-6:
        raise ValueError(f"Full-cache deterministic repeat audit failed: {repeat_difference.max():.3g}")

    input_hashes = {
        "provider_protocol_complete_sha256": sha256(args.provider_protocol / "complete.json"),
        "graph_manifest_complete_sha256": sha256(args.graph_manifest / "complete.json"),
        "smoke_complete_sha256": sha256(args.smoke / "complete.json"),
        "model_safetensors_sha256": sha256(args.snapshot / "model.safetensors"),
        "tokenizer_json_sha256": sha256(args.snapshot / "tokenizer.json"),
    }
    summary = {
        "model_id": MODEL_ID, "immutable_revision": REVISION,
        "parents": len(frame), "embedding_dimension": HIDDEN_SIZE,
        "dtype": "float32", "matrix_bytes": int(matrix.nbytes),
        "batch_size": args.batch_size, "maximum_tokens": int(frame.token_count.max()),
        "invalid_parents": 0, "unknown_token_parents": 0,
        "repeat_audit_parents": len(audit_indices),
        "repeat_audit_same_batch_membership_and_padding": True,
        "repeat_max_abs_difference": float(repeat_difference.max()),
        "remote_code_audit_passed": bool(code_audit.audit_pass.all()),
        "label_free": True, "validation_labels_read": False, "test_labels_read": False,
        "model_fitted": False, "model_selection_authorized": False,
    }
    with stage_output(args.output) as out:
        np.save(out / "embeddings_float32.npy", matrix)
        index = frame[[
            "molecule_id", "feature_index", "smiles_sha256", "token_count", "unknown_token_count"
        ]].copy()
        index.insert(0, "embedding_row", np.arange(len(index)))
        index["valid_embedding"] = True
        index["failure_reason"] = ""
        index.to_csv(out / "parent_embedding_index.csv", index=False)
        code_audit.to_csv(out / "remote_code_static_audit.csv", index=False)
        pd.DataFrame({
            "embedding_row": audit_indices,
            "molecule_id": frame.iloc[audit_indices].molecule_id.to_numpy(),
            "repeat_row_max_abs_difference": repeat_difference,
        }).to_csv(out / "repeatability_audit.csv", index=False)
        artifacts = []
        for name in ["config.json", "tokenizer.json", *REMOTE_CODE_FILES, "model.safetensors"]:
            path = args.snapshot / name
            artifacts.append({"file": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
        pd.DataFrame(artifacts).to_csv(out / "provider_artifact_hashes.csv", index=False)
        reloaded = np.load(out / "embeddings_float32.npy", mmap_mode="r")
        if reloaded.shape != matrix.shape or reloaded.dtype != np.float32 or not np.array_equal(reloaded, matrix):
            raise ValueError("Saved full embedding matrix failed exact reload verification")
        (out / "runtime_environment.json").write_text(json.dumps({
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": __import__("transformers").__version__, "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0),
        }, indent=2) + "\n", encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# MoLFormer parent embedding cache v1\n\n"
            "Immutable float32 frozen-encoder cache keyed by canonical parent molecule_id. It is label-free, uses no truncation, reads no split labels, and performs no model fitting. "
            "Downstream folds must join by molecule_id and fit all scaling/projection/model parameters within the training fold.\n",
            encoding="utf-8",
        )
        finish_stage(out, "molformer_parent_embedding_cache", inputs=input_hashes, partial=False, **summary)
    print(f"MoLFormer parent embedding cache: {args.output}")


if __name__ == "__main__":
    run_cli(main)
