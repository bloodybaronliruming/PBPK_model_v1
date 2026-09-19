"""Compose publication PNGs from the frozen Gate 8 R6 panel contract.

This is a layout-only renderer: it verifies frozen PNG hashes, preserves panel
aspect ratios, adds margins/panel letters/titles outside data layers, and saves
600-dpi PNGs. It does not modify source data marks, labels, colours, metrics,
or claims and reads no labels, predictions, or models.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, verify_stage


STAGE = "gate8_figure_composition"
R6_STAGE = "gate8_figure_contract"
R6_OUTPUT = "data/public_development/gate8_figure_contract_v1"
OUTPUT = "results/analysis/gate8_figure_composition_v1"
FONT = Path("/usr/share/fonts/truetype/DejaVuSans-Bold.ttf")
BG = (255, 255, 255)
TEXT = (24, 24, 24)
MARGIN = 90
HEADER = 130
PANEL_LABEL_HEIGHT = 72
GAP = 90


def fit_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image.copy()
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def text_font(size: int) -> ImageFont.FreeTypeFont:
    if not FONT.is_file():
        raise FileNotFoundError(f"Required reproducible font is missing: {FONT}")
    return ImageFont.truetype(str(FONT), size=size)


def title_height(title: str, font: ImageFont.FreeTypeFont, max_width: int) -> int:
    # Titles are short R6 contract strings; explicit guard prevents accidental clipping.
    if ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), title, font=font)[2] > max_width:
        raise ValueError(f"R6 caption title does not fit the frozen canvas: {title}")
    return HEADER


def draw_header(canvas: Image.Image, figure_id: str, title: str) -> None:
    draw = ImageDraw.Draw(canvas)
    font = text_font(48)
    text = f"{figure_id}. {title}"
    draw.text((MARGIN, 35), text, font=font, fill=TEXT)


def draw_panel_letter(canvas: Image.Image, letter: str, x: int, y: int) -> None:
    ImageDraw.Draw(canvas).text((x, y), letter, font=text_font(42), fill=TEXT)


def load_panel(row: pd.Series) -> Image.Image:
    path = Path(row.asset_path)
    if not path.is_file() or sha256(path) != row.asset_sha256:
        raise ValueError(f"Frozen R6 panel changed: {path}")
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.width != int(row.pixel_width) or image.height != int(row.pixel_height):
        raise ValueError(f"Frozen R6 panel dimension changed: {path}")
    return image


def vertical_layout(figure_id: str, title: str, rows: pd.DataFrame) -> tuple[Image.Image, list[dict]]:
    images = [(row, load_panel(row)) for _, row in rows.sort_values("panel").iterrows()]
    target_width = min(image.width for _, image in images)
    scaled = [(row, fit_width(image, target_width)) for row, image in images]
    width = target_width + 2 * MARGIN
    height = HEADER + sum(PANEL_LABEL_HEIGHT + image.height for _, image in scaled) + GAP * (len(scaled) - 1) + MARGIN
    canvas = Image.new("RGB", (width, height), BG)
    draw_header(canvas, figure_id, title)
    y = HEADER
    manifest = []
    for row, image in scaled:
        draw_panel_letter(canvas, row.panel, MARGIN, y + 10)
        top = y + PANEL_LABEL_HEIGHT
        canvas.paste(image, (MARGIN, top))
        manifest.append({"figure_id": figure_id, "panel": row.panel, "x": MARGIN, "y": top, "width": image.width, "height": image.height, "source_asset_sha256": row.asset_sha256})
        y = top + image.height + GAP
    return canvas, manifest


def grid_figure4(title: str, rows: pd.DataFrame) -> tuple[Image.Image, list[dict]]:
    expected = {"A", "B", "C", "D"}
    if set(rows.panel) != expected:
        raise ValueError("Figure 4 requires fixed A–D panels")
    images = {row.panel: (row, fit_width(load_panel(row), 4000)) for _, row in rows.iterrows()}
    row1_height = max(images[p][1].height for p in ("A", "B"))
    row2_height = max(images[p][1].height for p in ("C", "D"))
    panel_width = 4000
    width = 2 * MARGIN + 2 * panel_width + GAP
    height = HEADER + 2 * PANEL_LABEL_HEIGHT + row1_height + row2_height + GAP + MARGIN
    canvas = Image.new("RGB", (width, height), BG)
    draw_header(canvas, "Figure 4", title)
    placements = {"A": (MARGIN, HEADER), "B": (MARGIN + panel_width + GAP, HEADER), "C": (MARGIN, HEADER + PANEL_LABEL_HEIGHT + row1_height + GAP), "D": (MARGIN + panel_width + GAP, HEADER + PANEL_LABEL_HEIGHT + row1_height + GAP)}
    manifest = []
    for panel in ("A", "B", "C", "D"):
        row, image = images[panel]
        x, label_y = placements[panel]
        draw_panel_letter(canvas, panel, x, label_y + 10)
        top = label_y + PANEL_LABEL_HEIGHT
        canvas.paste(image, (x, top))
        manifest.append({"figure_id": "Figure 4", "panel": panel, "x": x, "y": top, "width": image.width, "height": image.height, "source_asset_sha256": row.asset_sha256})
    return canvas, manifest


def figure6_layout(title: str, rows: pd.DataFrame) -> tuple[Image.Image, list[dict]]:
    expected = {"A", "B", "C"}
    if set(rows.panel) != expected:
        raise ValueError("Figure 6 requires fixed A–C panels")
    by_panel = {row.panel: (row, load_panel(row)) for _, row in rows.iterrows()}
    top_width = 3300
    top = {panel: (row, fit_width(image, top_width)) for panel, (row, image) in by_panel.items() if panel in {"A", "B"}}
    bottom_row, bottom_source = by_panel["C"]
    bottom_width = 2 * top_width + GAP
    bottom = fit_width(bottom_source, bottom_width)
    top_height = max(image.height for _, image in top.values())
    width = 2 * MARGIN + bottom_width
    height = HEADER + PANEL_LABEL_HEIGHT + top_height + GAP + PANEL_LABEL_HEIGHT + bottom.height + MARGIN
    canvas = Image.new("RGB", (width, height), BG)
    draw_header(canvas, "Figure 6", title)
    manifest = []
    for panel, x in (("A", MARGIN), ("B", MARGIN + top_width + GAP)):
        row, image = top[panel]
        label_y = HEADER
        draw_panel_letter(canvas, panel, x, label_y + 10)
        y = label_y + PANEL_LABEL_HEIGHT
        canvas.paste(image, (x, y))
        manifest.append({"figure_id": "Figure 6", "panel": panel, "x": x, "y": y, "width": image.width, "height": image.height, "source_asset_sha256": row.asset_sha256})
    label_y = HEADER + PANEL_LABEL_HEIGHT + top_height + GAP
    draw_panel_letter(canvas, "C", MARGIN, label_y + 10)
    y = label_y + PANEL_LABEL_HEIGHT
    canvas.paste(bottom, (MARGIN, y))
    manifest.append({"figure_id": "Figure 6", "panel": "C", "x": MARGIN, "y": y, "width": bottom.width, "height": bottom.height, "source_asset_sha256": bottom_row.asset_sha256})
    return canvas, manifest


def compose(figure_id: str, title: str, rows: pd.DataFrame) -> tuple[Image.Image, list[dict]]:
    if figure_id == "Figure 4":
        return grid_figure4(title, rows)
    if figure_id == "Figure 6":
        return figure6_layout(title, rows)
    return vertical_layout(figure_id, title, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / OUTPUT
    r6 = root / R6_OUTPUT
    r6_meta = verify_stage(r6, R6_STAGE)
    required_flags = ("no_model_loaded", "no_labels_accessed", "no_prediction_rows_accessed", "no_metric_tables_accessed", "no_performance_metrics_computed", "no_plot_regenerated", "no_candidate_selection_changed")
    if not all(r6_meta.get(flag) is True for flag in required_flags):
        raise ValueError("R6 does not provide a frozen no-label figure contract")
    panel_path = r6 / "figure_panel_asset_manifest.csv"
    caption_path = r6 / "figure_caption_contract.csv"
    panels = pd.read_csv(panel_path)
    captions = pd.read_csv(caption_path)
    if len(panels) != 13 or len(captions) != 6 or set(captions.figure_id) != {f"Figure {i}" for i in range(1, 7)}:
        raise ValueError("R6 figure membership is not the frozen 6-figure/13-panel contract")
    inputs = {
        str((r6 / "complete.json").resolve()): sha256(r6 / "complete.json"),
        str(panel_path.resolve()): sha256(panel_path),
        str(caption_path.resolve()): sha256(caption_path),
    }
    if args.check_only:
        # Hash and panel dimension checks are deliberately executed even in preflight.
        for _, row in panels.iterrows():
            load_panel(row)
        print("Gate 8 R7 preflight: figures=6 frozen_panels=13 layout_only=yes 600dpi_output=yes; no labels, predictions, models, metric tables, or data-layer changes.")
        return
    with stage_output(output) as folder:
        render_rows = []
        layout_rows = []
        for figure_id in [f"Figure {i}" for i in range(1, 7)]:
            title = captions.loc[captions.figure_id.eq(figure_id), "english_caption_title"]
            if len(title) != 1:
                raise ValueError(f"Missing unique R6 caption title for {figure_id}")
            source_rows = panels.loc[panels.figure_id.eq(figure_id)].copy()
            image, layout = compose(figure_id, title.item(), source_rows)
            title_height(title.item(), text_font(48), image.width - 2 * MARGIN)
            filename = f"{figure_id.lower().replace(' ', '_')}_publication.png"
            destination = folder / filename
            image.save(destination, format="PNG", dpi=(600, 600), optimize=True)
            if not destination.is_file() or destination.stat().st_size == 0:
                raise RuntimeError(f"PNG rendering failed: {destination}")
            render_rows.append({
                "figure_id": figure_id, "output_png": filename, "output_sha256": sha256(destination),
                "pixel_width": image.width, "pixel_height": image.height, "dpi_x": 600, "dpi_y": 600,
                "panel_count": len(source_rows), "layout_only": True,
            })
            layout_rows.extend(layout)
        pd.DataFrame(render_rows).to_csv(folder / "publication_figure_manifest.csv", index=False)
        pd.DataFrame(layout_rows).to_csv(folder / "publication_figure_layout_manifest.csv", index=False)
        captions.to_csv(folder / "figure_caption_contract_copy.csv", index=False)
        (folder / "README.md").write_text(
            "# Gate 8 R7 publication figure composition\n\n"
            "Six 600-dpi PNGs were composed from the frozen R6 assets using layout-only operations: white canvas, external figure title, "
            "panel letters and aspect-ratio-preserving panel placement. Source PNG data layers, numerical content, colours and R6 claim "
            "boundaries were not modified.\n",
            encoding="utf-8",
        )
        finish_stage(
            folder, STAGE, inputs=inputs, partial=False, no_model_loaded=True, no_labels_accessed=True,
            no_prediction_rows_accessed=True, no_metric_tables_accessed=True, no_performance_metrics_computed=True,
            no_data_layer_modified=True, no_model_fitted=True, no_calibration=True, no_candidate_selection_changed=True,
            figure_count=6, source_panel_count=13, output_dpi=600, layout_only=True,
            source_contract_version="gate8_figure_contract_v1",
        )
    print(f"Gate 8 R7 publication figures: {output}")


if __name__ == "__main__":
    run_cli(main)
