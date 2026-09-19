#!/usr/bin/env python3
"""Download public PMC scanned-page images referenced by a saved article HTML page."""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import urllib.parse
import urllib.request
from pathlib import Path


IMAGE_RE = re.compile(
    r'<img\s+class="graphic"\s+src="([^"]+)"[^>]*\salt="([^"]+)"', re.IGNORECASE)
ALLOWED_HOST = "cdn.ncbi.nlm.nih.gov"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def page_images(html: str) -> list[tuple[str, str]]:
    result = []
    seen = set()
    for url, page in IMAGE_RE.findall(html):
        url = url.replace("&amp;", "&")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
            raise ValueError(f"非 PMC CDN 扫描页地址: {url}")
        key = (url, page.strip())
        if key not in seen:
            seen.add(key)
            result.append(key)
    if not result:
        raise ValueError("保存的 PMC HTML 中没有扫描页图片")
    return result


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()
    if not args.html.is_file():
        raise FileNotFoundError(args.html)
    images = page_images(args.html.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, (url, page) in enumerate(images, start=1):
        request = urllib.request.Request(url, headers={"User-Agent": "MTL-PMC-review/1.0"})
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            data = response.read()
        if not data.startswith(PNG_SIGNATURE):
            raise ValueError(f"PMC 扫描页不是 PNG: {url}")
        name = f"page_{position:02d}_{re.sub(r'[^A-Za-z0-9_-]+', '_', page)}.png"
        target = args.output_dir / name
        target.write_bytes(data)
        rows.append({"position": position, "printed_page": page, "file": name,
                     "source_url": url, "bytes": len(data), "sha256": sha256(data)})
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Downloaded {len(rows)} PMC scanned pages to {args.output_dir}")


if __name__ == "__main__":
    main()
