#!/usr/bin/env python3
"""Create a bounded, label-blind source-metadata audit queue for public-F candidates.

The queue deliberately reads bibliographic fields from the local ChEMBL ``docs``
table only.  It never accesses ``docs.abstract`` or any activity-value field.
Its title screen is a retrieval-routing aid, not an endpoint or label decision.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


FORBIDDEN = {
    "standard_value", "canonical_value", "raw_value", "target_value", "lower_bound", "upper_bound",
    "value", "mean", "median", "y",
}
METADATA_COLUMNS = [
    "doc_id", "journal", "year", "volume", "issue", "first_page", "last_page", "pubmed_id", "doi",
    "title", "doc_type", "authors",
]
SECONDARY_TITLE_PATTERN = re.compile(
    r"\b(qsar|model(?:s|ing)?|prediction|predicting|estimating|database|dataset|review|properties)\b", re.I
)


def normalise_identifier(value: object) -> str:
    text = str(value).strip().lower()
    # pandas represents an integer column with NULLs as e.g. ``17870541.0``.
    # Remove only that unambiguous storage artefact; do not coerce arbitrary IDs.
    return re.sub(r"^([0-9]+)\.0$", r"\1", text)


def classify_retrieval_route(metadata: pd.Series) -> tuple[str, str]:
    """Return a conservative routing signal based on bibliographic metadata alone."""
    title = str(metadata.get("title", ""))
    doc_type = str(metadata.get("doc_type", "")).upper()
    has_identifier = bool(str(metadata.get("doi", "")).strip() or str(metadata.get("pubmed_id", "")).strip())
    if doc_type in {"DATASET", "PATENT"} or not has_identifier:
        return (
            "bibliography_resolution_or_citation_backtracking",
            "Resolve a citable primary source before definition review; this metadata record is not itself sufficient provenance.",
        )
    if SECONDARY_TITLE_PATTERN.search(title):
        return (
            "citation_backtracking_before_primary_source_request",
            "Title suggests a model, review, or compiled source; inspect cited experimental provenance before requesting numeric evidence.",
        )
    return (
        "primary_source_definition_review_required",
        "Retrieve source methods/metadata to verify human population, parent analyte, continuous F semantics, and provenance.",
    )


def select_documents(documents: pd.DataFrame, max_p2_documents: int) -> pd.DataFrame:
    """Select all P1 sources and a bounded, deterministic set of P2 sources."""
    required = {"doc_id", "priority", "parents", "records", "scaffolds", "document_assays", "doi", "pubmed_id"}
    if not required <= set(documents.columns):
        raise ValueError("Source-aggregation document queue schema mismatch")
    if FORBIDDEN & set(documents.columns):
        raise ValueError("Metadata queue received a numeric endpoint column")
    p1 = documents.loc[documents.priority.eq("P1_metadata_IV_PO")].copy()
    p2 = documents.loc[documents.priority.eq("P2_metadata_PO")].copy()
    if p1.doc_id.duplicated().any() or p2.doc_id.duplicated().any():
        raise ValueError("A document appears more than once within a retrieval priority")
    p2 = p2.loc[~p2.doc_id.isin(p1.doc_id)].head(max_p2_documents)
    selected = pd.concat([p1, p2], ignore_index=True)
    if selected.empty or selected.doc_id.duplicated().any():
        raise ValueError("No unique P1/P2 source documents selected")
    selected.insert(0, "selection_rank", range(1, len(selected) + 1))
    selected.insert(1, "selection_rule", f"all_P1_plus_top_{max_p2_documents}_P2_by_existing_coverage_order")
    return selected


def read_document_metadata(database: Path, document_ids: list[str]) -> pd.DataFrame:
    """Read only explicitly enumerated, non-abstract bibliographic fields from ChEMBL."""
    if not database.is_file():
        raise FileNotFoundError(f"Missing local ChEMBL database: {database}")
    placeholders = ",".join("?" for _ in document_ids)
    query = f"SELECT {', '.join(METADATA_COLUMNS)} FROM docs WHERE doc_id IN ({placeholders})"
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        metadata = pd.read_sql_query(query, connection, params=[int(x) for x in document_ids])
    metadata = metadata.fillna("").astype(str)
    for identifier in ("doc_id", "pubmed_id"):
        metadata[identifier] = metadata[identifier].map(normalise_identifier)
    if len(metadata) != len(document_ids):
        found = set(metadata.doc_id)
        raise ValueError(f"ChEMBL docs metadata missing for doc_id(s): {sorted(set(document_ids) - found)}")
    if metadata.doc_id.duplicated().any():
        raise ValueError("ChEMBL docs metadata is not unique by doc_id")
    return metadata


def database_fingerprint(database: Path) -> dict:
    """Record a bounded reproducibility fingerprint without re-hashing a 29-GB release file.

    The released queue itself contains every field used from ``docs`` and is
    checksummed by ``finish_stage``.  Re-hashing the full third-party SQLite
    distribution at every small queue stage provides little additional
    protection while turning a metadata-only operation into a long job.
    """
    stat = database.stat()
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]
        docs_rows = connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        release_columns = [row[1] for row in connection.execute("PRAGMA table_info(chembl_release)")]
        releases = connection.execute(f"SELECT {', '.join(release_columns)} FROM chembl_release").fetchall()
    return {
        "path": str(database.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "sqlite_schema_version": schema_version, "sqlite_user_version": user_version,
        "docs_rows": docs_rows, "chembl_release_columns": release_columns,
        "chembl_release_rows": [list(row) for row in releases],
    }


def merge_metadata(selected: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    candidate = selected.copy().fillna("")
    candidate["doc_id"] = candidate.doc_id.astype(str)
    local = metadata.copy().fillna("").astype(str)
    merged = candidate.merge(local, on="doc_id", how="left", suffixes=("_candidate", "_local"), validate="one_to_one")
    for field in ("doi", "pubmed_id"):
        left, right = f"{field}_candidate", f"{field}_local"
        both = merged[left].map(normalise_identifier).ne("") & merged[right].map(normalise_identifier).ne("")
        conflict = both & merged[left].map(normalise_identifier).ne(merged[right].map(normalise_identifier))
        if conflict.any():
            raise ValueError(f"Candidate and local ChEMBL {field} disagree for doc_id(s): {merged.loc[conflict, 'doc_id'].tolist()}")
        merged[field] = merged[right].where(merged[right].ne(""), merged[left])
        merged = merged.drop(columns=[left, right])
    routes = merged.apply(classify_retrieval_route, axis=1, result_type="expand")
    merged[["retrieval_route", "retrieval_instruction"]] = routes
    ordered = [
        "selection_rank", "selection_rule", "doc_id", "priority", "document_assays", "records", "parents", "scaffolds",
        "doi", "pubmed_id", "journal", "year", "volume", "issue", "first_page", "last_page", "title", "doc_type", "authors",
        "definition_signals", "routes", "next_action", "retrieval_route", "retrieval_instruction",
    ]
    return merged.loc[:, ordered].sort_values("selection_rank", key=lambda values: values.astype(int), ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregation", type=Path, default=ROOT / "results/analysis/public_f_source_aggregation_v2")
    parser.add_argument("--chembl-db", type=Path, default=ROOT / "data/raw/chembl_37/chembl_37_sqlite/chembl_37.db")
    parser.add_argument("--max-p2-documents", type=int, default=11)
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/public_f_source_metadata_audit_queue_v2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.max_p2_documents < 1:
        raise ValueError("--max-p2-documents must be positive")
    document_queue = args.aggregation / "chembl_document_priority_queue_blinded.csv"
    required = [document_queue, args.aggregation / "complete.json", args.chembl_db]
    startup_self_check(required, output=None if args.check_only else args.output)
    verify_stage(args.aggregation, "public_f_source_aggregation")
    documents = pd.read_csv(document_queue, dtype=str, keep_default_na=False)
    selected = select_documents(documents, args.max_p2_documents)
    metadata = read_document_metadata(args.chembl_db, selected.doc_id.tolist())
    queue = merge_metadata(selected, metadata)
    database_identity = database_fingerprint(args.chembl_db)
    summary = {
        "selected_documents": len(queue),
        "selected_p1_documents": int(queue.priority.eq("P1_metadata_IV_PO").sum()),
        "selected_p2_documents": int(queue.priority.eq("P2_metadata_PO").sum()),
        "retrieval_route_counts": queue.retrieval_route.value_counts().sort_index().to_dict(),
        "numeric_values_read": False,
        "abstract_read": False,
        "database_fields_read": METADATA_COLUMNS,
        "chembl_database_fingerprint": database_identity,
        "selection_is_not_definition_admission": True,
    }
    if args.check_only:
        print(f"Public-F metadata audit queue valid: selected_documents={summary['selected_documents']}")
        return
    with stage_output(args.output) as out:
        queue.to_csv(out / "source_metadata_audit_queue_blinded.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "README.md").write_text(
            "# Public-F source metadata audit queue\n\n"
            "This is a bounded, label-blind retrieval queue: all P1 documents plus the first 11 P2 documents in the pre-existing "
            "coverage order. It reads only bibliographic fields from the local ChEMBL `docs` table and explicitly does not query "
            "the abstract or any activity value. `retrieval_route` is a title/doc-type routing signal, never an F-definition, "
            "human-population, or label-admission decision. A document may enter B1 only after a separate definition audit; values "
            "remain hidden until a separately versioned B2 numerical lock.\n",
            encoding="utf-8",
        )
        finish_stage(out, "public_f_source_metadata_audit_queue", inputs={
            "aggregation_complete_sha256": sha256(args.aggregation / "complete.json"),
            "chembl_sqlite_fingerprint": database_identity,
        }, **summary, partial=False)
    print(f"Public-F source metadata audit queue: {args.output}")


if __name__ == "__main__":
    run_cli(main)
