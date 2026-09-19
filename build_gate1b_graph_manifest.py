#!/usr/bin/env python3
"""Build a canonical-parent graph manifest for Gate 1B without fitting models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from neural_models import ATOM_DIM, BOND_DIM, molecular_graph
from pipeline_common import ROOT, finish_stage, run_cli, sha256, stable_id, stage_output, startup_self_check, verify_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl-protocol", type=Path, default=ROOT / "data/public_development/stl_benchmark_protocol_v1")
    parser.add_argument("--multimodal-protocol", type=Path, default=ROOT / "data/public_development/gate1b_multimodal_protocol_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_development/gate1b_graph_manifest_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    required = [args.stl_protocol / "complete.json", args.stl_protocol / "feature_row_manifest.csv",
                args.multimodal_protocol / "complete.json"]
    startup_self_check(required, output=None if args.check_only else args.output)
    stl_meta = verify_stage(args.stl_protocol, "stl_benchmark_protocol")
    multi_meta = verify_stage(args.multimodal_protocol, "gate1b_multimodal_input_protocol")
    if stl_meta.get("test_labels_read") or multi_meta.get("test_labels_read"):
        raise ValueError("Graph manifest cannot consume a test-open upstream stage")
    inputs = {
        "stl_protocol_complete_sha256": sha256(args.stl_protocol / "complete.json"),
        "multimodal_protocol_complete_sha256": sha256(args.multimodal_protocol / "complete.json"),
    }
    if args.check_only and (args.output / "complete.json").exists():
        meta = verify_stage(args.output, "gate1b_graph_manifest")
        if meta.get("inputs") != inputs:
            raise ValueError("Published graph manifest no longer matches its inputs")
        print(f"Gate 1B graph manifest valid: parents={meta['parents']} zero_bond={meta['zero_bond_parents']}")
        return
    parents = pd.read_csv(args.stl_protocol / "feature_row_manifest.csv")
    rows = []
    for row in parents.itertuples(index=False):
        frozen_smiles = str(row.canonical_parent_smiles)
        if stable_id(frozen_smiles) != row.molecule_id:
            raise ValueError(f"Frozen canonical-parent string hash mismatch: {row.molecule_id}")
        mol = Chem.MolFromSmiles(frozen_smiles)
        if mol is None:
            raise ValueError(f"Invalid canonical parent: {row.molecule_id}")
        # Some frozen parent strings retain explicit hydrogen atoms.  A graph
        # encoder must not treat those serialization artifacts as a separate
        # atom modality, so create one deterministic H-suppressed graph view
        # while preserving the original parent hash as the membership key.
        graph_mol = Chem.RemoveHs(mol)
        graph_smiles = Chem.MolToSmiles(graph_mol, isomericSmiles=False)
        graph_roundtrip = Chem.MolFromSmiles(graph_smiles)
        if graph_roundtrip is None or Chem.MolToInchiKey(mol) != Chem.MolToInchiKey(graph_roundtrip):
            raise ValueError(f"H-suppressed graph identity mismatch: {row.molecule_id}")
        atoms, bonds, src, dst, reverse = molecular_graph(graph_smiles)
        if atoms.shape[1] != ATOM_DIM or bonds.shape[1] != BOND_DIM:
            raise ValueError("Graph feature schema mismatch")
        if not np.isfinite(atoms).all() or not np.isfinite(bonds).all():
            raise ValueError(f"Non-finite graph feature: {row.molecule_id}")
        if not (len(src) == len(dst) == len(reverse) == len(bonds)):
            raise ValueError(f"Directed-edge array mismatch: {row.molecule_id}")
        rows.append({
            "molecule_id": row.molecule_id,
            "feature_index": int(row.feature_index),
            "frozen_parent_smiles": frozen_smiles,
            "graph_smiles": graph_smiles,
            "atoms": len(atoms),
            "directed_bonds": len(bonds),
            "undirected_bonds": len(bonds) // 2,
            "zero_bond_graph": len(bonds) == 0,
        })
    manifest = pd.DataFrame(rows).sort_values("feature_index")
    if not manifest.feature_index.tolist() == list(range(len(manifest))):
        raise ValueError("Graph feature indices are not contiguous and aligned")
    summary = {
        "parents": len(manifest),
        "atom_features": ATOM_DIM,
        "bond_features": BOND_DIM,
        "zero_bond_parents": int(manifest.zero_bond_graph.sum()),
        "max_atoms": int(manifest.atoms.max()),
        "max_undirected_bonds": int(manifest.undirected_bonds.max()),
        "model_fitted": False,
        "test_labels_read": False,
    }
    if args.check_only:
        print(f"Gate 1B graph manifest valid: parents={len(manifest)} zero_bond={summary['zero_bond_parents']}")
        return
    with stage_output(args.output) as out:
        manifest.to_csv(out / "graph_manifest.csv", index=False)
        (out / "graph_schema.json").write_text(json.dumps({
            "atom_features": ATOM_DIM,
            "bond_features": BOND_DIM,
            "directed_edges": True,
            "atom_schema": "project neural_models.atom_features v1",
            "bond_schema": "single|double|triple|aromatic|conjugated|in_ring",
            "canonicalization": "membership inherits frozen STL parent hash; graph view removes explicit H and stereochemistry",
            "rdkit_version": rdBase.rdkitVersion,
        }, indent=2) + "\n", encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Gate 1B graph manifest\n\nCanonical-parent graph coverage and schema only. No labels, models, validation, or test data are read.\n",
            encoding="utf-8",
        )
        finish_stage(out, "gate1b_graph_manifest", inputs=inputs, **summary, partial=False)
    print(f"Gate 1B graph manifest: {args.output}")


if __name__ == "__main__":
    run_cli(main)
