"""
scripts/download_images.py

Extracts ALL images from the OpenStax PDF into data/images/.

All content is CC BY 4.0 (OpenStax).

Usage:
    python scripts/download_images.py                        # auto-detect PDF
    python scripts/download_images.py --pdf PATH/TO/FILE.pdf # specific PDF
    python scripts/download_images.py --min-size 250         # min px on either side (default 250)
    python scripts/download_images.py --verify               # list extracted images
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── PDF image extraction ──────────────────────────────────────────────────────


def extract_all_images_from_pdf(pdf_path: Path, images_dir: Path, min_size: int = 250) -> int:
    """
    Extract every image from the PDF into images_dir.

    Images are saved as pdf_p{page}_{idx}.{ext}, identical images are deduped by
    content hash, and an image_metadata.json index is written. Images smaller
    than min_size px on either side are skipped (icons, rules, etc.).
    Returns the number of images saved.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  pypdf not installed — run: pip install pypdf")
        return 0

    print(f"\n  Reading PDF: {pdf_path.name} ({pdf_path.stat().st_size / (1024 * 1024):.0f}MB)")

    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        print(f"  Pages: {total_pages}")
    except Exception as e:
        print(f"  Cannot read PDF: {e}")
        return 0

    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting all images -> {images_dir} ...")

    seen_hashes: set[str] = set()
    per_page_counter: dict[int, int] = {}
    index: list[dict] = []
    saved = 0
    duplicates = 0

    for page_num in range(total_pages):
        try:
            page = reader.pages[page_num]
            resources = page.get("/Resources")
            if not resources:
                continue
            resources = resources.get_object() if hasattr(resources, "get_object") else resources
            xobjects = resources.get("/XObject")
            if not xobjects:
                continue
            xobjects = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects

            for name in xobjects:
                try:
                    obj = xobjects[name].get_object()
                    if obj.get("/Subtype") != "/Image":
                        continue
                    w = int(obj.get("/Width", 0))
                    h = int(obj.get("/Height", 0))
                    if w < min_size or h < min_size:  # skip tiny icons / rules
                        continue

                    data = obj.get_data()
                    digest = hashlib.md5(data).hexdigest()
                    if digest in seen_hashes:
                        duplicates += 1
                        continue
                    seen_hashes.add(digest)

                    img_filter = obj.get("/Filter", "")
                    ext = "jpg" if img_filter == "/DCTDecode" else "png"

                    page = page_num + 1
                    per_page_counter[page] = per_page_counter.get(page, 0) + 1
                    idx = per_page_counter[page]

                    filename = f"pdf_p{page:04d}_{idx}.{ext}"
                    out_path = images_dir / filename
                    out_path.write_bytes(data)
                    index.append(
                        {
                            "filename": filename,
                            "page": page,
                            "width": w,
                            "height": h,
                            "size_kb": out_path.stat().st_size // 1024,
                            "source": "OpenStax A&P 2e",
                            "license": "CC BY 4.0",
                        }
                    )
                    saved += 1
                except Exception:
                    continue

        except Exception:
            continue

        if (page_num + 1) % 200 == 0:
            print(f"    Scanned {page_num + 1}/{total_pages} pages, {saved} images saved...")

    out = ROOT / "data" / "image_metadata.json"
    out.write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"  Saved {saved} unique images ({duplicates} duplicates skipped)")
    print(f"  Index: data/image_metadata.json ({len(index)} entries)")
    return saved


# ── Main ──────────────────────────────────────────────────────────────────────


def find_pdf(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    candidates = sorted(
        ROOT.glob("data/raw/**/*.pdf"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    for c in candidates:
        if c.stat().st_size > 10_000_000:  # > 10MB = real PDF
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract all anatomy images for socratOT")
    parser.add_argument("--pdf", type=str, default=None)
    parser.add_argument(
        "--min-size",
        type=int,
        default=250,
        help="Minimum width/height in px to keep an image (default 250)",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    images_dir = ROOT / "data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  socratOT — Anatomy Image Acquisition")
    print(f"{'=' * 60}")

    if args.verify:
        existing = sorted(images_dir.glob("*"))
        print(f"  Files in data/images/: {len(existing)}")
        for f in existing[:10]:
            print(f"    {f.name} ({f.stat().st_size // 1024}KB)")
        return

    pdf_path = find_pdf(args.pdf)
    if not (pdf_path and pdf_path.exists()):
        print("\n  No PDF found.")
        print("  Tip: run with --pdf PATH/TO/anatomy.pdf, or place a PDF in data/raw/")
        return

    n = extract_all_images_from_pdf(pdf_path, images_dir, min_size=args.min_size)

    print(f"\n{'=' * 60}")
    print(f"  Done! {n} images in data/images/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
