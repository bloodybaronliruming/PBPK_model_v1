#!/usr/bin/env python3
"""Bind Jia R3b scoring rules and frozen model identities without reading labels."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from pipeline_common import (ROOT, configure_logging, dump_json, finish_stage, run_cli, sha256,
                             stage_output, startup_self_check, verify_stage)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r3a",type=Path,default=ROOT/"models/jia2025_author_like_r3a_v1")
    p.add_argument("--n0",type=Path,default=ROOT/"results/analysis/jia2025_n0_audit_v1")
    p.add_argument("--evaluator",type=Path,default=ROOT/"scripts/evaluate_jia2025_author_like_r3b.py")
    p.add_argument("--output",type=Path,default=ROOT/"data/public_development/jia2025_author_like_r3b_protocol_v5")
    a=p.parse_args(); startup_self_check([a.r3a/"complete.json",a.n0/"complete.json",a.evaluator],output=a.output); configure_logging(ROOT,"register_jia2025_r3b")
    r3a=verify_stage(a.r3a,"jia2025_author_like_r3a_traincv_and_model_freeze"); n0=verify_stage(a.n0,"jia2025_n0_reproducibility_audit")
    if r3a.get("author_test_predictions") or r3a.get("author_test_metrics") or n0.get("provisional_reproduction_grade")!="B": raise ValueError("Input lifecycle/grade invalid")
    models=pd.read_csv(a.r3a/"frozen_model_manifest.csv"); models["artifact"] = models.artifact.map(lambda x: str(a.r3a.relative_to(ROOT) / x)); members=pd.read_csv(a.n0/"author_split_membership_hashed.csv")
    test=members.loc[members.author_split.eq("test"),["endpoint","parent_id"]].rename(columns={"parent_id":"parent_hash"}).sort_values(["endpoint","parent_hash"])
    if models.groupby("endpoint").size().to_dict()!={"CL":18,"Fu":18,"VDss":18} or test.groupby("endpoint").size().to_dict()!={"CL":176,"Fu":632,"VDss":176}: raise ValueError("Unexpected frozen model/test membership counts")
    inputs={str(x.resolve()):sha256(x) for x in [a.r3a/"complete.json",a.r3a/"frozen_model_manifest.csv",a.n0/"complete.json",a.n0/"author_split_membership_hashed.csv",a.evaluator]}
    with stage_output(a.output) as out:
        models.to_csv(out/"model_hashes.csv",index=False); test.to_csv(out/"author_test_membership_hashed.csv",index=False)
        dump_json(out/"scoring_rules.json",{"grade":"B_author_like","cohorts":["author_native_record","author_parent_purged_sensitivity"],"paper_primary":{"Fu":"paper_svr_merged","CL":"paper_consensus_15","VDss":"paper_consensus_15"},"consensus":"arithmetic mean of 15 log10 paper-model predictions","metrics":["R2_log","MAE_log","RMSE_log","GMFE","within_2fold"],"selection_after_score":"prohibited"})
        dump_json(out/"author_test_lifecycle.json",{"N0_status":"technical_label_presence_accessed_not_scored","r3a_predictions":False,"r3a_metrics":False,"permitted_next":"one descriptive score only","rerun_after_score":"prohibited"})
        (out/"protocol_report.md").write_text("# Jia R3b one-time descriptive scoring protocol\n\nThis package binds 54 R3a frozen models, N0 hashed author-test membership, native and parent-purged cohorts, paper consensus and a no-selection/no-rerun rule. Registration reads no labels and generates no predictions.\n",encoding="utf-8")
        finish_stage(out,"jia2025_author_like_r3b_scoring_protocol",inputs=inputs,models_bound=len(models),author_test_membership_parents=len(test),author_test_scored=False,author_test_predictions=False,author_test_metrics=False,protocol_grade="B_author_like",technical_recovery_of="r3b_v3_pre_prediction_parent_hash_mismatch",partial=False)
    print(f"Jia 2025 R3b scoring protocol: {a.output}")
if __name__=="__main__": run_cli(main)
