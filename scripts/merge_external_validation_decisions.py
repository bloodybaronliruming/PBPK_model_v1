"""Append or safely revise audited external-validation decisions in a new registry."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT


DECISIONS = {"accepted", "rejected", "pending_clinical_methods", "pending_fulltext"}
REQUIRED_COLUMNS = [
    "candidate_id", "activity_id", "molecule_id", "source_doi", "eligibility_decision",
    "primary_source_doi", "source_table_or_page", "route_evidence",
    "terminal_phase_evidence", "parent_systemic_evidence", "extracted_value",
    "extracted_unit", "exclusion_reason", "evidence_note", "reviewer", "review_date",
]


def validate_decisions(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(REQUIRED_COLUMNS)
    if missing or extra:
        raise ValueError(f"{name} 注册表列不一致 missing={sorted(missing)} extra={sorted(extra)}")
    frame = frame[REQUIRED_COLUMNS].copy().fillna("")
    if frame.candidate_id.astype(str).str.strip().eq("").any() or frame.candidate_id.duplicated().any():
        raise ValueError(f"{name} 含空或重复 candidate_id")
    invalid = set(frame.eligibility_decision) - DECISIONS
    if invalid:
        raise ValueError(f"{name} 含无效裁定: {sorted(invalid)}")
    accepted = frame.eligibility_decision.eq("accepted")
    rejected = frame.eligibility_decision.eq("rejected")
    for column in ["molecule_id", "primary_source_doi", "source_table_or_page", "route_evidence",
                   "terminal_phase_evidence", "parent_systemic_evidence", "extracted_unit",
                   "evidence_note", "reviewer", "review_date"]:
        if frame.loc[accepted, column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} accepted 行缺少 {column}")
    numeric = pd.to_numeric(frame.loc[accepted, "extracted_value"], errors="coerce")
    if numeric.isna().any() or numeric.le(0).any():
        raise ValueError(f"{name} accepted 行必须有正数 extracted_value")
    if frame.loc[accepted, "extracted_unit"].str.lower().ne("h").any():
        raise ValueError(f"{name} accepted 行当前只接受小时单位")
    if frame.loc[rejected, "exclusion_reason"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} rejected 行缺少 exclusion_reason")
    if frame.loc[~accepted, "extracted_value"].astype(str).str.strip().ne("").any():
        raise ValueError(f"{name} 非 accepted 行不得含 extracted_value")
    return frame


def merge(base: pd.DataFrame, additions: pd.DataFrame,
          replace_existing: bool = False) -> pd.DataFrame:
    base = validate_decisions(base, "base")
    additions = validate_decisions(additions, "additions")
    overlap = set(base.candidate_id) & set(additions.candidate_id)
    if replace_existing:
        missing = set(additions.candidate_id) - set(base.candidate_id)
        if missing:
            raise ValueError(f"修订裁定不在基准注册表: {sorted(missing)}")
        identity_columns = ["activity_id", "molecule_id", "source_doi"]
        base_indexed = base.set_index("candidate_id")
        for row in additions.itertuples(index=False):
            original = base_indexed.loc[row.candidate_id]
            expected = tuple(str(original[column]).strip().lower()
                             for column in identity_columns)
            observed = tuple(str(getattr(row, column)).strip().lower()
                             for column in identity_columns)
            if observed != expected:
                raise ValueError(
                    f"修订不得改变候选身份字段: {row.candidate_id} "
                    f"observed={observed} expected={expected}")
        revised = base.set_index("candidate_id")
        revisions = additions.set_index("candidate_id")
        revised.loc[revisions.index, REQUIRED_COLUMNS[1:]] = revisions[REQUIRED_COLUMNS[1:]]
        return revised.reset_index()[REQUIRED_COLUMNS]
    if overlap:
        raise ValueError(f"新增裁定与基准 candidate_id 重复: {sorted(overlap)}")
    return pd.concat([base, additions], ignore_index=True)


def validate_against_candidates(decisions: pd.DataFrame, candidates: pd.DataFrame) -> None:
    """Require identity fields to match the label-blind candidate index exactly."""
    required = {"candidate_id", "activity_id", "molecule_id", "doi"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate index 缺少列: {sorted(missing)}")
    index = candidates[list(required)].copy().fillna("")
    if index.candidate_id.astype(str).str.strip().eq("").any() or index.candidate_id.duplicated().any():
        raise ValueError("candidate index 含空或重复 candidate_id")
    indexed = index.set_index("candidate_id")
    for row in decisions.itertuples(index=False):
        if row.candidate_id not in indexed.index:
            raise ValueError(f"裁定不在 candidate index: {row.candidate_id}")
        candidate = indexed.loc[row.candidate_id]
        expected = {
            "activity_id": str(candidate.activity_id).strip(),
            "molecule_id": str(candidate.molecule_id).strip(),
            "source_doi": str(candidate.doi).strip().lower(),
        }
        observed = {
            "activity_id": str(row.activity_id).strip(),
            "molecule_id": str(row.molecule_id).strip(),
            "source_doi": str(row.source_doi).strip().lower(),
        }
        if observed != expected:
            raise ValueError(
                f"裁定身份字段与 candidate index 不一致: {row.candidate_id} "
                f"observed={observed} expected={expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=ROOT / "results/analysis/thalf_external_validation_decisions_v6.csv")
    parser.add_argument("--additions", type=Path,
                        default=ROOT / "results/analysis/pkdb_external_validation_decisions_v1.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/thalf_external_validation_decisions_v7.csv")
    parser.add_argument("--candidate-index", type=Path,
                        help="可选的盲态候选 CSV；校验新增裁定的 candidate/activity/molecule/DOI 身份")
    parser.add_argument("--replace-existing", action="store_true",
                        help="仅修订基准中已有 candidate_id；保持行数、顺序和身份字段不变")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.replace_existing and not args.candidate_index:
        parser.error("--replace-existing 必须同时提供 --candidate-index")
    for path in [args.base, args.additions]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.check_only and args.output.exists():
        raise FileExistsError(f"禁止覆盖已有注册表: {args.output}")
    additions = pd.read_csv(args.additions, keep_default_na=False)
    additions = validate_decisions(additions, "additions")
    if args.candidate_index:
        if not args.candidate_index.is_file():
            raise FileNotFoundError(args.candidate_index)
        validate_against_candidates(additions, pd.read_csv(args.candidate_index, keep_default_na=False))
    merged = merge(pd.read_csv(args.base, keep_default_na=False), additions,
                   replace_existing=args.replace_existing)
    counts = merged.eligibility_decision.value_counts().to_dict()
    accepted_molecules = merged.loc[
        merged.eligibility_decision.eq("accepted"), "molecule_id"].nunique()
    if args.check_only:
        print(f"Registry check passed: rows={len(merged)} decisions={counts} "
              f"accepted_molecules={accepted_molecules}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: rows={len(merged)} decisions={counts} "
          f"accepted_molecules={accepted_molecules}")


if __name__ == "__main__":
    main()
