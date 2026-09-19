"""Build a concise, evidence-linked English Gate 8 manuscript outline.

The stage verifies only completed Gate 8 registry/table/figure/supplement
stages. It writes a narrative outline and claim map without reading labels,
prediction rows, models, metrics, or MMPK numerical material.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


STAGE = "gate8_manuscript_outline"
OUTPUT = "进度/投稿草稿/gate8_manuscript_outline_v1"
SOURCES = [
    ("R0 evidence registry", "data/public_development/gate8_manuscript_evidence_registry_v2", "gate8_manuscript_evidence_registry"),
    ("Table 1 context", "data/public_development/gate8_table1_schema_v1", "gate8_table1_schema"),
    ("Table 2 boundaries", "data/public_development/gate8_table2_literature_boundaries_v1", "gate8_table2_literature_boundaries"),
    ("Table 3 internal evidence", "data/public_development/gate8_table3_strict_internal_v1", "gate8_table3_strict_internal"),
    ("Table 4 frozen evidence", "data/public_development/gate8_table4_frozen_evidence_v1", "gate8_table4_frozen_evidence"),
    ("Supplement manifest", "data/public_development/gate8_supplement_manifest_v1", "gate8_supplement_manifest"),
    ("Figure contract", "data/public_development/gate8_figure_contract_v1", "gate8_figure_contract"),
    ("Publication figures", "results/analysis/gate8_figure_composition_v1", "gate8_figure_composition"),
]


OUTLINE = """# Working manuscript outline

## Candidate title

**Leakage-Controlled, Endpoint-Specific Modelling of Human DMPK Properties: Matched Generalization Tests and Reproducible Evidence Delivery**

Short title: **Evidence-governed human DMPK modelling**

## Structured abstract (draft scaffold)

**Background.** Human DMPK endpoints differ in assay definition, route, source structure, physical range and evidence maturity; a single model or a single reported score can therefore be misleading.

**Methods.** We assembled endpoint-specific human tasks, preserved parent/scaffold/source/study and derivation boundaries, and evaluated strong single-task baselines before matched tests of multitask learning, cross-species transfer and experimental-physchem augmentation. Candidate, test-lifecycle and reproducibility records were frozen before manuscript assembly.

**Results.** Endpoint-specific strong STL models remained the research reference. Matched analyses documented where alternative architectures or cross-species inputs did not provide a stable replacement, and the preregistered B6 experimental-physchem route was stopped for all eligible endpoints. Frozen-test, AD and calibration diagnostics are reported as closed descriptive evidence, not as post-test selection inputs. Absolute oral F remains exploratory because validation is insufficient.

**Conclusions.** The principal contribution is an auditable endpoint-specific evidence system: data-definition governance, leakage control, transparent negative results and reproducible delivery are reported alongside model performance. Claims are limited by source overlap, endpoint-specific generalization, uncalibrated uncertainty and restricted data rights.

## Introduction

1. State why fu, microsomal CLint, Caco-2 Papp A→B, systemic IV CL, IV VDss and terminal IV t½ cannot be treated as interchangeable labels.
2. Motivate endpoint-specific modelling and strict grouping against parent/scaffold/source/study/derivation leakage.
3. Define the study question: whether more complex, multitask, cross-species or experimental-physchem routes provide evidence strong enough to replace a strong STL reference under matched protocols.
4. State the contribution as an evidence-governed workflow, not a universal architecture claim.

## Methods

### Data governance and tasks

- Define the seven task keys, units, assay/route context and F status in Table 1.
- Describe reliability/source rules, canonical parent identity and the distinction between source truth tables and derived training views.
- State that F is retained as status-only exploratory rather than being promoted to a confirmatory numeric model.

### Leakage control and evaluation lifecycle

- Describe fixed parent/scaffold/source/study/derivation isolation and train-fold-only preprocessing.
- State that frozen tests are closed after their planned use and cannot be used for candidate replacement, calibration or threshold selection.
- Describe A/B/C evidence levels; Jia author-like rows are B, published Jia values are C, and no A-level direct replication is claimed.

### Modelling and matched ablations

- Present strong STL and endpoint-specific references first.
- Describe matched P1, multitask, cross-species and B6 ablations as bounded decisions rather than unbounded model search.
- State that B6 was governed by preregistered gates and retained as a negative result.

### Reproducibility and reporting

- Describe versioned artifacts, hashes, no-label technical inference/reload checks, data/model cards and manuscript asset contracts.
- State that technical replay validates software delivery only; it is not predictive validation or a unified numerical release.

## Results

### 1. Endpoint definitions and evidence lanes

Use Table 1 and Figure 1 to show task-specific definitions, records/parents, reference lanes, test lifecycle and limitations.

### 2. Strict internal reference and negative architecture evidence

Use Table 3 and Figure 2 for matched STL evidence. Use Figure 3 for multitask conflict as an associative diagnostic. Use Figure 5 to report the B6 0/6 advance outcome without reopening that route.

### 3. Controlled cross-species transfer

Use Figure 4 to report endpoint-specific matched human-only versus transfer results. Retain both human-only and strong STL comparators in every interpretation.

### 4. Frozen diagnostic and delivery evidence

Use Table 4 and Figure 6 to report frozen-test lifecycle, AD/similarity, calibration limitations, source-overlap restrictions and clean-environment replay. Do not combine historical and later research-reference lanes.

### 5. Literature and data-rights context

Use Table 2 to place Jia published C-level background and project author-like B-level results in separate rows. State that MMPK remains internal-only and contributes no public numerical comparison.

## Discussion

1. Emphasize that endpoint-specific strong baselines are a valid outcome when complex models do not clear registered replacement criteria.
2. Discuss negative transfer and negative B6 findings as decision-relevant evidence, not failed experiments to hide.
3. Discuss cross-species transfer only as controlled, endpoint-specific evidence; avoid a universal species-transfer conclusion.
4. Explain why frozen tests, source overlap, low-similarity regions and uncalibrated intervals limit clinical or independent-external claims.
5. State that the dose/route-conditioned NCA line is separate from the structure-only standard-condition manuscript.

## Limitations

- F lacks sufficient independent validation and remains status-only exploratory.
- Terminal t½ has documented TDC/Obach ecosystem overlap; it is not an independent external validation claim.
- Most numeric endpoints lack calibrated predictive intervals; technical ensemble spread is not coverage calibration.
- Some historical frozen assets and later strict train-CV references are different evidence lanes and must remain separated.
- Supplementary foldwise/detail material is not assembled because R0 did not register the required assets; MMPK numerical material is excluded pending data rights.

## Declarations and availability

- Code/environment/provenance: cite the reproducibility manifest and clean-reload verification.
- Data availability: describe source-specific rights and do not redistribute MMPK-derived data, predictions or models.
- Conflict of interest, author contributions and funding: complete after project administration review.
"""


def claim_map() -> pd.DataFrame:
    rows = [
        ("ABS-1", "Abstract", "Endpoint-specific evidence-governed workflow", "not_comparison", "Table 1; Figure 1; R0 registry", "Do not claim a universal model winner."),
        ("ABS-2", "Abstract", "B6 was stopped rather than selectively continued", "B", "Table 3; Figure 5; S6", "Report 0/6 only as the frozen B6 decision."),
        ("ABS-3", "Abstract", "F remains exploratory", "not_comparison", "Table 1; Table 4", "No numeric F model or test-performance claim."),
        ("MET-1", "Methods", "Endpoint definitions and units", "not_comparison", "Table 1", "Do not merge assay or route definitions."),
        ("MET-2", "Methods", "Parent/scaffold/source/study/derivation leakage controls", "not_comparison", "Figure 1; R0 registry", "Describe frozen protocol; do not infer unregistered fold membership."),
        ("MET-3", "Methods", "A/B/C comparison hierarchy", "not_comparison", "Table 2; Table 3", "Only A-level permits direct reproduction/overperformance language."),
        ("RES-1", "Results", "Strict internal reference and matched P1 evidence", "B", "Table 3; Figure 2; S5", "Compare only within the stated comparison_id."),
        ("RES-2", "Results", "Multitask conflict diagnostic", "B", "Figure 3", "Association only; no causal mechanism claim."),
        ("RES-3", "Results", "Cross-species ablations", "B", "Figure 4", "Retain matched human-only and strong STL controls."),
        ("RES-4", "Results", "B6 negative ablation", "B", "Table 3; Figure 5; S6", "Do not reopen the stopped B6 family."),
        ("RES-5", "Results", "Frozen diagnostic/AD/calibration status", "not_comparison", "Table 4; Figure 6", "Closed tests cannot be used for selection or calibration."),
        ("RES-6", "Results", "Jia context", "B/C separated", "Table 2", "Do not rank B author-like and C published rows together."),
        ("RES-7", "Results", "MMPK restriction", "internal_only", "Table 2 boundary; S9", "No public numerical comparison or redistributable output."),
        ("DIS-1", "Discussion", "Endpoint-specific models may be preferable", "B", "Table 3; Figures 2–5", "Do not generalize beyond matched protocols."),
        ("DIS-2", "Discussion", "Terminal t½ overlap limitation", "not_comparison", "Table 4; Figure 6", "No independent external-validation language."),
        ("DIS-3", "Discussion", "Technical reproducibility", "not_comparison", "Table 4; S7", "Not predictive validation or a unified numerical release."),
        ("LIM-1", "Limitations", "F scope and validation gap", "not_comparison", "Table 1; Table 4", "Status-only endpoint."),
        ("LIM-2", "Limitations", "Supplement and rights gaps", "not_comparison/internal_only", "S8; S9", "Disclose unassembled foldwise detail and restricted MMPK material."),
    ]
    return pd.DataFrame(rows, columns=["claim_id", "manuscript_section", "permitted_statement", "evidence_level", "bound_evidence", "claim_guardrail"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / OUTPUT
    inputs = {}
    for label, relative, expected_stage in SOURCES:
        folder = root / relative
        verify_stage(folder, expected_stage)
        complete = folder / "complete.json"
        inputs[str(complete.resolve())] = sha256(complete)
    map_frame = claim_map()
    if len(map_frame) != 18 or map_frame.claim_id.duplicated().any():
        raise ValueError("Manuscript claim map must contain 18 unique claim records")
    if args.check_only:
        print("Gate 8 R8 preflight: verified_stages=8 outline_sections=8 claim_records=18; no labels, prediction rows, models, metrics, or MMPK numerical material read.")
        return
    with stage_output(output) as folder:
        (folder / "manuscript_outline.md").write_text(OUTLINE, encoding="utf-8")
        map_frame.to_csv(folder / "section_evidence_map.csv", index=False)
        pd.DataFrame(SOURCES, columns=["source_role", "source_path", "expected_stage"]).to_csv(folder / "verified_source_stages.csv", index=False)
        (folder / "README.md").write_text(
            "# Gate 8 R8 manuscript outline\n\n"
            "This concise English manuscript scaffold is bound to completed Gate 8 tables, figures and Supplement boundaries. "
            "It introduces no numerical result and does not read labels, predictions, models, metric tables or MMPK numerical material. "
            "Every permitted narrative claim is indexed in `section_evidence_map.csv`; the outline must not be expanded into a stronger claim without a matching evidence update.\n",
            encoding="utf-8",
        )
        finish_stage(
            folder, STAGE, inputs=inputs, partial=False, no_model_loaded=True, no_labels_accessed=True,
            no_prediction_rows_accessed=True, no_metric_tables_accessed=True, no_performance_metrics_computed=True,
            no_model_fitted=True, no_calibration=True, no_candidate_selection_changed=True,
            no_mmpk_numeric_values_accessed=True, outline_section_count=8, claim_record_count=len(map_frame),
            verified_stage_count=len(SOURCES), source_registry_version="gate8_manuscript_evidence_registry_v2",
        )
    print(f"Gate 8 R8 manuscript outline: {output}")


if __name__ == "__main__":
    run_cli(main)
