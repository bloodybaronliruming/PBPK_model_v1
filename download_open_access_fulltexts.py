#!/usr/bin/env python3
"""Download legally accessible full-text PDFs for papers listed in a CSV.

The script intentionally uses open-access metadata and public URLs only. It
does not use Sci-Hub, bypass CAPTCHAs, evade paywalls, or automate logins.
Rows that appear to require institutional access or manual verification are
written to a separate report.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
DEFAULT_USER_AGENT = "MTL-fulltext-fetcher/1.0"


@dataclass
class Candidate:
    url: str
    source: str
    license: str = ""
    evidence: str = ""


class PdfLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "meta" and values.get("name", "").lower() == "citation_pdf_url":
            self.urls.append(values.get("content", ""))
        elif tag.lower() == "link" and "pdf" in values.get("type", "").lower():
            self.urls.append(values.get("href", ""))
        elif tag.lower() == "a":
            href = values.get("href", "")
            if href.lower().split("?", 1)[0].endswith(".pdf"):
                self.urls.append(href)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download open-access PDFs for DOI/PubMed rows in a CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Input CSV with doi/pubmed_id columns.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/literature/fulltexts"),
        help="Directory for PDFs and reports.",
    )
    parser.add_argument("--doi-column", default="doi", help="CSV column containing DOI.")
    parser.add_argument(
        "--id-column",
        default="doc_id",
        help="CSV column used in output filenames when present.",
    )
    parser.add_argument(
        "--email",
        default="",
        help="Contact email for Unpaywall/Crossref-style API etiquette.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum rows to process; 0 means all rows.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to wait between external metadata/download attempts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even when the target PDF already exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in an existing fulltext_manifest.csv.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def normalize_doi(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    match = DOI_RE.search(value)
    if not match:
        return ""
    return match.group(0).rstrip(".,;)")


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_")[:180] or "paper"


def request_json(url: str, headers: dict[str, str], timeout: int) -> dict:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (TimeoutError, socket.timeout) as exc:
        raise TimeoutError("metadata_timeout") from exc


def query_unpaywall(doi: str, email: str, headers: dict[str, str], timeout: int) -> list[Candidate]:
    if not email:
        return []
    url = "https://api.unpaywall.org/v2/{doi}?email={email}".format(
        doi=urllib.parse.quote(doi, safe=""),
        email=urllib.parse.quote(email),
    )
    data = request_json(url, headers, timeout)
    candidates: list[Candidate] = []
    best = data.get("best_oa_location") or {}
    locations = [best] + [loc for loc in data.get("oa_locations") or [] if loc != best]
    for loc in locations:
        pdf_url = loc.get("url_for_pdf") or ""
        landing_url = loc.get("url") or ""
        if pdf_url:
            candidates.append(
                Candidate(
                    url=pdf_url,
                    source="unpaywall",
                    license=loc.get("license") or "",
                    evidence=loc.get("host_type") or "",
                )
            )
        elif landing_url and loc.get("is_best"):
            candidates.append(
                Candidate(
                    url=landing_url,
                    source="unpaywall_landing",
                    license=loc.get("license") or "",
                    evidence=loc.get("host_type") or "",
                )
            )
    return dedupe_candidates(candidates)


def query_openalex(doi: str, headers: dict[str, str], timeout: int) -> list[Candidate]:
    url = "https://api.openalex.org/works/doi:{doi}".format(
        doi=urllib.parse.quote(doi, safe="/:")
    )
    data = request_json(url, headers, timeout)
    candidates: list[Candidate] = []
    for loc in data.get("locations") or []:
        pdf_url = (loc.get("pdf_url") or "").strip()
        landing_url = (loc.get("landing_page_url") or "").strip()
        source_name = ((loc.get("source") or {}).get("display_name") or "openalex").strip()
        if pdf_url:
            candidates.append(Candidate(pdf_url, "openalex", evidence=source_name))
        elif landing_url and loc.get("is_oa"):
            candidates.append(Candidate(landing_url, "openalex_landing", evidence=source_name))
    return dedupe_candidates(candidates)


def doi_landing_candidate(doi: str) -> Candidate:
    return Candidate("https://doi.org/" + urllib.parse.quote(doi, safe="/"), "doi_landing")


def dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        key = candidate.url.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def download_pdf(url: str, path: Path, headers: dict[str, str], timeout: int) -> tuple[bool, str]:
    req = urllib.request.Request(
        url,
        headers={
            **headers,
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
    except urllib.error.HTTPError as exc:
        return False, f"http_{exc.code}"
    except urllib.error.URLError as exc:
        return False, f"url_error:{exc.reason}"
    except (TimeoutError, socket.timeout):
        return False, "timeout"

    if not data.startswith(b"%PDF"):
        if "text/html" in content_type.lower():
            return False, "html_or_landing_page"
        return False, f"not_pdf:{content_type or 'unknown_content_type'}"

    path.write_bytes(data)
    return True, "downloaded"


def landing_pdf_candidates(
    candidate: Candidate, headers: dict[str, str], timeout: int
) -> list[Candidate]:
    landing_url = candidate.url
    old_pmc = re.search(r"ncbi\.nlm\.nih\.gov/pmc/articles/(?:PMC)?(\d+)", landing_url, re.IGNORECASE)
    direct_candidates: list[Candidate] = []
    if old_pmc:
        landing_url = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{old_pmc.group(1)}/"
        direct_candidates.append(
            Candidate(
                urllib.parse.urljoin(landing_url, "pdf/"),
                f"{candidate.source}_pmc_pdf",
                evidence="PubMed Central",
            )
        )
    req = urllib.request.Request(landing_url, headers={**headers, "Accept": "text/html,*/*;q=0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            final_url = response.geturl()
            data = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, socket.timeout):
        return direct_candidates

    if data.startswith(b"%PDF") or "pdf" in content_type.lower():
        return dedupe_candidates(
            direct_candidates
            + [Candidate(final_url, f"{candidate.source}_resolved", evidence="direct_pdf")]
        )
    if "html" not in content_type.lower() and not data.lstrip().startswith(b"<"):
        return direct_candidates

    parser = PdfLinkParser()
    parser.feed(data.decode("utf-8", errors="replace"))
    return dedupe_candidates(
        direct_candidates
        + [
            Candidate(
                urllib.parse.urljoin(final_url, url),
                f"{candidate.source}_resolved",
                license=candidate.license,
                evidence=candidate.evidence,
            )
            for url in parser.urls
            if url
        ]
    )


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str], doi_column: str, id_column: str) -> str:
    doi = normalize_doi(row.get(doi_column, ""))
    row_id = row.get(id_column) or row.get("pubmed_id") or ""
    return f"{row_id}|{doi}"


def main() -> int:
    args = parse_args()
    rows = read_rows(args.csv_path)
    if args.limit:
        rows = rows[: args.limit]

    pdf_dir = args.output_dir / "pdf"
    report_dir = args.output_dir / "reports"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    if args.email:
        headers["From"] = args.email

    manifest: list[dict[str, str]] = []
    manual: list[dict[str, str]] = []
    base_fields = list(rows[0].keys()) if rows else []
    extra_fields = [
        "normalized_doi",
        "status",
        "pdf_path",
        "chosen_url",
        "source",
        "reason",
        "candidate_urls",
    ]
    manifest_path = report_dir / "fulltext_manifest.csv"
    manual_path = report_dir / "manual_required.csv"
    processed_keys: set[str] = set()
    if args.resume and manifest_path.exists():
        manifest = read_rows(manifest_path)
        manual = [row for row in manifest if row.get("status") in {"manual_required", "missing_doi"}]
        processed_keys = {row_key(row, "normalized_doi", args.id_column) for row in manifest}
        processed_keys.update(row_key(row, args.doi_column, args.id_column) for row in manifest)
        print(f"Resuming from {manifest_path}: {len(processed_keys)} rows already processed.")

    for index, row in enumerate(rows, start=1):
        if args.resume and row_key(row, args.doi_column, args.id_column) in processed_keys:
            continue

        doi = normalize_doi(row.get(args.doi_column, ""))
        row_id = row.get(args.id_column) or row.get("pubmed_id") or str(index)
        base = safe_name(f"{row_id}_{doi.replace('/', '_') if doi else 'no_doi'}")
        pdf_path = pdf_dir / f"{base}.pdf"
        status = "missing_doi"
        chosen_url = ""
        source = ""
        reason = ""
        candidates: list[Candidate] = []

        if doi:
            if pdf_path.exists() and not args.overwrite:
                status = "exists"
                reason = "already_downloaded"
            else:
                for provider in (
                    lambda: query_unpaywall(doi, args.email, headers, args.timeout),
                    lambda: query_openalex(doi, headers, args.timeout),
                ):
                    try:
                        candidates.extend(provider())
                    except Exception as exc:  # noqa: BLE001 - metadata failures should not stop batch runs.
                        reason = f"metadata_error:{type(exc).__name__}:{exc}"
                    time.sleep(args.sleep)

                candidates = dedupe_candidates(candidates + [doi_landing_candidate(doi)])
                download_candidates: list[Candidate] = []
                for candidate in candidates:
                    if candidate.url.lower().split("?", 1)[0].endswith(".pdf") or (
                        "pdf" in candidate.url.lower()
                    ):
                        download_candidates.append(candidate)
                    else:
                        download_candidates.extend(
                            landing_pdf_candidates(candidate, headers, args.timeout)
                        )
                        time.sleep(args.sleep)

                for candidate in dedupe_candidates(download_candidates):
                    ok, download_reason = download_pdf(candidate.url, pdf_path, headers, args.timeout)
                    time.sleep(args.sleep)
                    if ok:
                        status = "downloaded"
                        chosen_url = candidate.url
                        source = candidate.source
                        reason = download_reason
                        break
                    reason = download_reason
                else:
                    status = "manual_required"
                    if not reason:
                        reason = "no_open_pdf_found"

        record = {
            **row,
            "normalized_doi": doi,
            "status": status,
            "pdf_path": str(pdf_path if pdf_path.exists() else ""),
            "chosen_url": chosen_url,
            "source": source,
            "reason": reason,
            "candidate_urls": json.dumps([candidate.__dict__ for candidate in candidates], ensure_ascii=False),
        }
        manifest.append(record)
        if status in {"manual_required", "missing_doi"}:
            manual.append(record)

        write_csv(manifest_path, base_fields + extra_fields, manifest)
        write_csv(manual_path, base_fields + extra_fields, manual)
        print(f"[{index}/{len(rows)}] {doi or row_id}: {status} {reason}")

    write_csv(manifest_path, base_fields + extra_fields, manifest)
    write_csv(manual_path, base_fields + extra_fields, manual)

    summary = {
        "input": str(args.csv_path),
        "total_rows": len(rows),
        "downloaded": sum(1 for item in manifest if item["status"] == "downloaded"),
        "existing": sum(1 for item in manifest if item["status"] == "exists"),
        "manual_required": len(manual),
        "output_dir": str(args.output_dir),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
