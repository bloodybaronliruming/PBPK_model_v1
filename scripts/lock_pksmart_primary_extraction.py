#!/usr/bin/env python3
"""Lock PKSmart primary-source population extraction and one prespecified formal aggregate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


POPULATION_COLUMNS = ["candidate_id", "eligibility_registry_sha256", "cohort_id", "population", "n", "t_half_h",
                      "summary_statistic", "include_in_formal_aggregate", "source_table_or_page", "extraction_note",
                      "reviewer", "review_date"]
AGGREGATE_COLUMNS = ["candidate_id", "eligibility_registry_sha256", "aggregate_population_definition", "selected_cohort_ids",
                     "extracted_value", "extracted_unit", "aggregation_group", "aggregation_method", "source_table_or_page",
                     "extraction_note", "reviewer", "review_date"]


def validate(population: pd.DataFrame, aggregate: pd.DataFrame, eligibility: pd.DataFrame, lock_hash: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(population.columns) != set(POPULATION_COLUMNS) or set(aggregate.columns) != set(AGGREGATE_COLUMNS):
        raise ValueError("PKSmart extraction files do not match the registered schemas")
    population = population[POPULATION_COLUMNS].copy().fillna("")
    aggregate = aggregate[AGGREGATE_COLUMNS].copy().fillna("")
    accepted = set(eligibility.candidate_id)
    if set(population.candidate_id) != accepted or set(aggregate.candidate_id) != accepted:
        raise ValueError("PKSmart extraction must cover exactly the accepted eligibility candidates")
    if population.cohort_id.duplicated().any() or aggregate.candidate_id.duplicated().any():
        raise ValueError("PKSmart extraction contains duplicate cohorts or aggregates")
    if population.eligibility_registry_sha256.ne(lock_hash).any() or aggregate.eligibility_registry_sha256.ne(lock_hash).any():
        raise ValueError("PKSmart extraction must reference the current eligibility lock hash")
    for field in ["n", "t_half_h"]:
        value = pd.to_numeric(population[field], errors="coerce")
        if value.isna().any() or value.le(0).any() or not np.isfinite(value).all():
            raise ValueError(f"PKSmart population extraction has invalid {field}")
    values = pd.to_numeric(aggregate.extracted_value, errors="coerce")
    if values.isna().any() or values.le(0).any() or aggregate.extracted_unit.str.lower().ne("h").any():
        raise ValueError("PKSmart aggregate values must be positive hours")
    required = ["population", "summary_statistic", "source_table_or_page", "extraction_note", "reviewer", "review_date"]
    if any(population[field].eq("").any() for field in required):
        raise ValueError("PKSmart population extraction has incomplete provenance")
    required = ["aggregate_population_definition", "selected_cohort_ids", "aggregation_group", "aggregation_method",
                "source_table_or_page", "extraction_note", "reviewer", "review_date"]
    if any(aggregate[field].eq("").any() for field in required):
        raise ValueError("PKSmart aggregate extraction has incomplete provenance")
    for row in aggregate.itertuples(index=False):
        group = population.loc[population.candidate_id.eq(row.candidate_id)].copy()
        selected = set(str(row.selected_cohort_ids).split("|"))
        chosen = group.loc[group.cohort_id.isin(selected)]
        if set(chosen.cohort_id) != selected or not chosen.include_in_formal_aggregate.astype(bool).all():
            raise ValueError(f"PKSmart aggregate selection is inconsistent: {row.candidate_id}")
        if group.loc[~group.cohort_id.isin(selected), "include_in_formal_aggregate"].astype(bool).any():
            raise ValueError(f"PKSmart nonselected cohort marked formal: {row.candidate_id}")
        weighted = np.average(pd.to_numeric(chosen.t_half_h), weights=pd.to_numeric(chosen.n))
        if not np.isclose(float(row.extracted_value), weighted, rtol=0, atol=0.005):
            raise ValueError(f"PKSmart aggregate differs from its declared sample-size-weighted mean: {row.candidate_id}")
    return population, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligibility-lock", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_eligibility_lock_v1")
    parser.add_argument("--population", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_review_batch_v1/population_extraction_completed_v1.csv")
    parser.add_argument("--aggregate", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_review_batch_v1/formal_aggregate_completed_v1.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pksmart_primary_source_extraction_lock_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify_stage(args.eligibility_lock, "pksmart_primary_source_eligibility_lock")
    registry = args.eligibility_lock / "eligibility_registry_locked.csv"
    startup_self_check([registry, args.population, args.aggregate], output=None if args.check_only else args.output)
    lock_hash = sha256(registry)
    population, aggregate = validate(pd.read_csv(args.population, keep_default_na=False),
                                     pd.read_csv(args.aggregate, keep_default_na=False),
                                     pd.read_csv(registry, keep_default_na=False), lock_hash)
    if args.check_only:
        print(f"PKSmart primary-source extraction check passed: formal_molecules={len(aggregate)} supporting_cohorts={len(population)}")
        return
    with stage_output(args.output) as out:
        population.to_csv(out / "population_extraction_locked.csv", index=False)
        aggregate.to_csv(out / "formal_aggregate_locked.csv", index=False)
        (out / "README.md").write_text(
            "# PKSmart primary-source numerical extraction\n\n"
            "The formal external set receives one molecule-level aggregate per candidate. For trilaciclib, only the "
            "two matched healthy-control cohorts are included, with a sample-size-weighted arithmetic mean. Hepatic "
            "impairment cohorts are retained as supporting evidence and do not multiply the formal score denominator.\n",
            encoding="utf-8")
        finish_stage(out, "pksmart_primary_source_extraction_lock", inputs={
            "eligibility_registry_sha256": lock_hash, "population_input_sha256": sha256(args.population),
            "aggregate_input_sha256": sha256(args.aggregate),
        }, formal_molecules=len(aggregate), supporting_cohorts=len(population), partial=False)
    print(f"PKSmart primary-source extraction lock: {args.output}")


if __name__ == "__main__":
    run_cli(main)
