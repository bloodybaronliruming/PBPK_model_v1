#!/usr/bin/env python3
"""Render the label-blinded PKSmart candidate funnel as an interactive HTML fragment.

The visualization is generated from immutable stage summaries and a small
manually-audited IV-product lead manifest.  It does not load, parse, print, or
embed a PKSmart endpoint value.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, run_cli, startup_self_check, verify_stage


LEAD_COLUMNS = [
    "candidate_id", "chembl_id", "chembl_pref_name", "identity_mapping_basis", "official_label_url",
    "primary_source_locator", "primary_source_status", "route_note", "reviewer", "review_date",
]


def raw_row_count(path: Path) -> int:
    """Count CSV records without parsing the endpoint column or retaining its values."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.reader(stream)) - 1


def validate_leads(frame: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if set(frame.columns) != set(LEAD_COLUMNS):
        raise ValueError("IV product lead manifest columns do not match the locked schema")
    frame = frame[LEAD_COLUMNS].fillna("").copy()
    if frame.candidate_id.eq("").any() or frame.candidate_id.duplicated().any():
        raise ValueError("IV product lead manifest has an empty or duplicate candidate_id")
    if set(frame.primary_source_status) - {"primary_article_located", "primary_article_not_located"}:
        raise ValueError("Unexpected IV product lead source status")
    if (~frame.official_label_url.str.startswith("https://dailymed.nlm.nih.gov/")).any():
        raise ValueError("IV product lead manifest must cite a DailyMed label URL")
    indexed = mapping.set_index("candidate_id")
    if not set(frame.candidate_id) <= set(indexed.index):
        raise ValueError("IV product lead does not occur in the structure-isolated mapping")
    for row in frame.itertuples(index=False):
        mapped = indexed.loc[row.candidate_id]
        if str(mapped.chembl_id) != row.chembl_id or str(mapped.chembl_pref_name) != row.chembl_pref_name:
            raise ValueError(f"IV product lead identity mismatch: {row.candidate_id}")
        if mapped.mapping_status != row.identity_mapping_basis:
            raise ValueError(f"IV product lead mapping basis mismatch: {row.candidate_id}")
    return frame


def counts(raw: Path, queue: Path, mapping_dir: Path, leads_path: Path) -> dict[str, int]:
    mapping = pd.read_csv(mapping_dir / "candidate_mapping_blinded.csv", keep_default_na=False)
    if not mapping.endpoint_values_hidden.astype(bool).all():
        raise ValueError("Mapping unexpectedly exposes endpoint values")
    summary = pd.read_csv(queue / "screening_summary.csv", keep_default_na=False).set_index("screening_status")
    independent = int(summary.loc["primary_source_provenance_required", "records"])
    source_rows = int(__import__("json").loads((queue / "complete.json").read_text(encoding="utf-8"))["source_rows"])
    named = int(mapping.mapping_status.isin(["exact_chembl37", "neutralized_parent_chembl37"]).sum())
    leads = validate_leads(pd.read_csv(leads_path, keep_default_na=False), mapping)
    return {
        "public_rows": raw_row_count(raw), "human_rows": source_rows, "independent": independent,
        "named": named, "iv_product_leads": len(leads), "formal_labels": 0,
    }


def render(values: dict[str, int]) -> str:
    stage_rows = [
        ("Public external CSV rows", values["public_rows"], "muted"),
        ("Human half-life rows (label-blinded)", values["human_rows"], "primary"),
        ("After v15/TDC structure exclusion", values["independent"], "primary"),
        ("Named retrieval candidates", values["named"], "green"),
        ("Documented IV product leads", values["iv_product_leads"], "orange"),
        ("Formal external labels", values["formal_labels"], "destructive"),
    ]
    # JSON serialisation keeps the colour status as a string in JavaScript;
    # unquoted identifiers here would cause a ReferenceError at render time.
    stages = json.dumps(stage_rows, ensure_ascii=False)
    return f'''<section id="pksmart-candidate-funnel" aria-labelledby="pksmart-title">
  <h2 id="pksmart-title">PKSmart candidate triage: evidence flow and inclusion gates</h2>
  <div class="viz-controls" aria-label="View">
    <button class="btn btn-primary" type="button" id="show-funnel" aria-pressed="true">Candidate flow</button>
    <button class="btn" type="button" id="show-gates" aria-pressed="false">Inclusion gates</button>
  </div>
  <div id="funnel-panel" role="region" aria-label="Candidate flow">
    <svg id="funnel-svg" role="img" aria-labelledby="funnel-title funnel-desc" style="width:100%;height:auto"></svg>
    <p class="text-small text-muted">All numbers are candidate counts; no PKSmart half-life value is read or displayed.</p>
  </div>
  <div id="gates-panel" role="region" aria-label="Formal external-validation inclusion gates" hidden>
    <svg id="gates-svg" role="img" aria-labelledby="gates-title gates-desc" style="width:100%;height:auto"></svg>
    <p class="text-small text-muted">An IV product lead only prioritizes primary-study retrieval; it cannot bypass any gate.</p>
  </div>
  <p id="selection" class="sr-only" aria-live="polite"></p>
</section>
<script>
(() => {{
  const funnelButton = document.getElementById('show-funnel');
  const gatesButton = document.getElementById('show-gates');
  const funnelPanel = document.getElementById('funnel-panel');
  const gatesPanel = document.getElementById('gates-panel');
  const selection = document.getElementById('selection');
  const fg = 'var(--foreground)', muted = 'var(--muted-foreground)', border = 'var(--border)';
  const color = {{muted: muted, primary: 'var(--viz-series-1)', green: 'var(--viz-series-3)', orange: 'var(--viz-series-2)', destructive: 'var(--destructive)'}};
  function base(id, title, desc) {{
    const svg = document.getElementById(id); svg.setAttribute('viewBox', '0 0 700 330');
    svg.innerHTML = `<title id="${{id === 'funnel-svg' ? 'funnel-title' : 'gates-title'}}">${{title}}</title><desc id="${{id === 'funnel-svg' ? 'funnel-desc' : 'gates-desc'}}">${{desc}}</desc>`;
    return svg;
  }}
  function addText(svg, x, y, value, attrs='') {{ svg.insertAdjacentHTML('beforeend', `<text x="${{x}}" y="${{y}}" fill="${{fg}}" font-size="14" ${{attrs}}>${{value}}</text>`); }}
  function funnel() {{
    const svg = base('funnel-svg', 'PKSmart candidate flow', 'Candidate counts after structure isolation and identity mapping.');
    const stages = [
      {stages}
    ];
    const max = stages[0][1];
    stages.forEach((stage, i) => {{ const y = 18 + i * 50, width = Math.max(14, 500 * stage[1] / max); svg.insertAdjacentHTML('beforeend', `<rect x="170" y="${{y}}" width="${{width}}" height="30" rx="2" fill="${{color[stage[2]]}}" opacity="${{stage[1] === 0 ? '.28' : '.82'}}"></rect>`); addText(svg, 10, y + 20, stage[0]); addText(svg, 680, y + 20, String(stage[1]), 'text-anchor="end" font-weight="500"'); }});
    svg.insertAdjacentHTML('beforeend', `<line x1="170" y1="320" x2="670" y2="320" stroke="${{border}}" stroke-width="1"></line><text x="170" y="315" fill="${{muted}}" font-size="12">Candidate count (width scaled to public CSV rows)</text>`);
  }}
  function gates() {{
    const svg = base('gates-svg', 'Formal external-validation inclusion gates', 'All five evidence gates are required before a candidate becomes a formal label.');
    const data = [['Structure and source isolation','No overlap with v15, TDC, or accepted sources'],['Row-level primary-source locator','Article, clinical report, or a pre-registered regulatory track'],['Human direct IV administration','No oral, IM, animal, or inferred substitute'],['Systemic parent analyte','No metabolite, prodrug, or conjugate substitution'],['Terminal phase and point estimate','Defined method, page/table locator, and h unit']];
    data.forEach((gate, i) => {{ const x = 30 + i * 132, y = 45 + (i % 2) * 104, fill = i < 2 ? color.primary : color.green; svg.insertAdjacentHTML('beforeend', `<circle cx="${{x + 24}}" cy="${{y + 24}}" r="20" fill="${{fill}}" opacity=".82"></circle><text x="${{x + 24}}" y="${{y + 29}}" fill="var(--primary-foreground)" font-size="14" text-anchor="middle">${{i + 1}}</text><line x1="${{x + 47}}" y1="${{y + 24}}" x2="${{x + 121}}" y2="${{y + 24}}" stroke="${{border}}" stroke-width="2"></line>`); addText(svg, x, y + 60, gate[0], 'font-weight="500"'); svg.insertAdjacentHTML('beforeend', `<foreignObject x="${{x}}" y="${{y + 67}}" width="122" height="50"><div xmlns="http://www.w3.org/1999/xhtml" style="color:${{muted}};font-size:12px;line-height:1.3">${{gate[1]}}</div></foreignObject>`); }});
    svg.insertAdjacentHTML('beforeend', `<path d="M35 276 H625" stroke="${{border}}" stroke-width="1"></path><text x="35" y="304" fill="${{fg}}" font-size="14">All five gates are required before entry into the locked formal external-validation registry.</text>`);
  }}
  function show(which) {{ const isFunnel = which === 'funnel'; funnelPanel.hidden = !isFunnel; gatesPanel.hidden = isFunnel; funnelButton.classList.toggle('btn-primary', isFunnel); gatesButton.classList.toggle('btn-primary', !isFunnel); funnelButton.setAttribute('aria-pressed', String(isFunnel)); gatesButton.setAttribute('aria-pressed', String(!isFunnel)); selection.textContent = isFunnel ? 'Candidate flow shown.' : 'Formal external-validation inclusion gates shown.'; }}
  funnel(); gates(); funnelButton.addEventListener('click', () => show('funnel')); gatesButton.addEventListener('click', () => show('gates'));
}})();
</script>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path,
                        default=ROOT / "data/external/pksmart_external_candidate_v1/raw/External_test_315.csv")
    parser.add_argument("--queue", type=Path,
                        default=ROOT / "results/analysis/pksmart_external_candidate_queue_v1")
    parser.add_argument("--mapping", type=Path,
                        default=ROOT / "results/analysis/pksmart_external_chembl_mapping_v2")
    parser.add_argument("--iv-product-leads", type=Path,
                        default=ROOT / "data/external/pksmart_external_candidate_v1/iv_product_leads_v1.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify_stage(args.queue, "pksmart_external_candidate_queue")
    verify_stage(args.mapping, "pksmart_candidate_chembl_mapping")
    startup_self_check([args.raw, args.queue / "screening_summary.csv", args.mapping / "candidate_mapping_blinded.csv",
                        args.iv_product_leads], output=None if args.check_only else args.output)
    values = counts(args.raw, args.queue, args.mapping, args.iv_product_leads)
    if args.check_only:
        print(values)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(values), encoding="utf-8")
    print(f"Wrote reproducible label-blinded visualization: {args.output}")


if __name__ == "__main__":
    run_cli(main)
