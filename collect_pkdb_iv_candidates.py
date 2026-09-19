"""从 PK-DB 官方 API 保存 IV 快照并生成不含 PK 数值的候选研究表。"""
from __future__ import annotations

import argparse
import io
import json
import logging
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from pipeline_common import (ROOT, configure_logging, dump_json, finish_stage, run_cli,
                             sha256, stage_output, startup_self_check)


BASE_URL = "https://pk-db.com/api/v1"


def fetch(url: str, timeout: int = 180) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "MTL-model-PKDB-audit/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def build_blinded_candidates(studies: pd.DataFrame, groups: pd.DataFrame,
                             individuals: pd.DataFrame, interventions: pd.DataFrame) -> pd.DataFrame:
    def human_studies(frame: pd.DataFrame) -> set[str]:
        required = {"study_sid", "measurement_type", "choice"}
        if not required <= set(frame.columns):
            return set()
        mask = (frame.measurement_type.fillna("").str.lower().eq("species")
                & frame.choice.fillna("").str.lower().eq("homo sapiens"))
        return set(frame.loc[mask, "study_sid"].astype(str))

    human = human_studies(groups) | human_studies(individuals)
    iv = interventions.loc[interventions.route.fillna("").str.lower().eq("iv")].copy()
    candidates = studies.loc[studies.sid.astype(str).isin(human)].copy()
    iv_summary = (iv.groupby("study_sid", as_index=False)
                  .agg(iv_substances=("substance", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
                       iv_interventions=("intervention_pk", "nunique")))
    candidates = candidates.merge(iv_summary, left_on="sid", right_on="study_sid", how="inner",
                                  validate="one_to_one")
    candidates["half_life_output_status"] = "unresolved_api_outputs_empty"
    candidates["eligibility_status"] = "primary_source_screening_required"
    candidates["endpoint_values_hidden"] = True
    prohibited = {"value", "mean", "median", "min", "max", "sd", "se", "cv"}
    if prohibited & set(candidates.columns):
        raise RuntimeError("PK-DB 盲态候选包含数值列")
    return candidates


def run(output: Path, base_url: str = BASE_URL) -> None:
    startup_self_check(output=output)
    statistics_url = f"{base_url}/statistics/"
    filter_url = f"{base_url}/filter/?" + urllib.parse.urlencode({
        "interventions__route_sid": "iv", "download": "true",
    })
    statistics_raw = fetch(statistics_url)
    archive_raw = fetch(filter_url)
    statistics = json.loads(statistics_raw)
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"PK-DB 下载 ZIP 损坏: {bad}")
        names = set(archive.namelist())
        required = {"studies.csv", "groups.csv", "individuals.csv", "interventions.csv", "outputs.csv"}
        if not required <= names:
            raise ValueError(f"PK-DB 下载缺少表: {sorted(required - names)}")
        frames = {name: pd.read_csv(archive.open(name)) for name in required}
    candidates = build_blinded_candidates(frames["studies.csv"], frames["groups.csv"],
                                          frames["individuals.csv"], frames["interventions.csv"])
    outputs_available = not frames["outputs.csv"].empty

    with stage_output(output) as out:
        raw = out / "raw"
        raw.mkdir()
        (raw / "statistics.json").write_bytes(statistics_raw)
        (raw / "pkdb_iv_download.zip").write_bytes(archive_raw)
        candidates.to_csv(out / "candidate_studies_blinded.csv", index=False)
        dump_json(out / "api_status.json", {
            "base_url": base_url,
            "statistics": statistics,
            "iv_archive_tables": {name: len(frame) for name, frame in frames.items()},
            "outputs_available": outputs_available,
            "known_limitation": ("The production outputs endpoint/download is empty; candidate studies "
                                 "must be screened in primary sources before endpoint values are acquired.")
                                if not outputs_available else "",
        })
        finish_stage(out, "pkdb_iv_candidate_snapshot", inputs={
            "statistics_url": statistics_url, "filter_url": filter_url,
        }, api_version=statistics.get("version"), studies=statistics.get("study_count"),
                     iv_studies=len(frames["studies.csv"]), human_iv_candidate_studies=len(candidates),
                     outputs_available=outputs_available, label_blinded=True, partial=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    configure_logging(args.root, "collect_pkdb_iv_candidates")
    local_repo = args.root / "data/pkpb_datbase/data/raw/pkdb"
    startup_self_check([local_repo / "INSTALLATION.md"])
    if args.check_only:
        stats = json.loads(fetch(f"{args.base_url}/statistics/", timeout=30))
        logging.info("PK-DB API 可用，版本 %s，studies=%s。", stats.get("version"), stats.get("study_count"))
        return
    run(args.output or args.root / "data/external/pkdb_iv_candidates_v1", args.base_url)


if __name__ == "__main__":
    run_cli(main)
