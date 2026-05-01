"""Download all static datasets for the Boston Real Estate Value Analyzer.

Usage:
    python scripts/download_datasets.py [--output-dir data/raw]

Downloads:
    1. Boston Property Assessment FY2026 (Analyze Boston)
    2. Boston Parcels 2025 CSV (Analyze Boston)
    3. Zillow ZHVI by ZIP Code
    4. Redfin ZIP-level market tracker
    5. MBTA GTFS (extracts stops.txt)
    6. BPD Crime Incident Reports (2023-2025)
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ---------------------------------------------------------------------------
# Dataset registry — each entry maps a friendly name to its download URL,
# output filename, and optional post-processing.
# ---------------------------------------------------------------------------

DATASETS: list[dict] = [
    {
        "name": "Boston Property Assessment FY2026",
        "url": (
            "https://data.boston.gov/dataset/"
            "e02c44d2-3c64-459c-8fe2-e1ce5f38a035/resource/"
            "ee73430d-96c0-423e-ad21-c4cfb54c8961/download/"
            "fy2026-property-assessment-data_12_23_2025.csv"
        ),
        "filename": "boston_property_assessment_fy2026.csv",
    },
    {
        "name": "Zillow ZHVI by ZIP Code",
        "url": (
            "https://files.zillowstatic.com/research/public_csvs/zhvi/"
            "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
        ),
        "filename": "zillow_zhvi_by_zip.csv",
    },
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "RealEstateTracker/1.0 (BU CS506 Student Project)"
)


def download_file(url: str, dest: Path) -> None:
    """Download a URL to a local file with a progress indicator."""
    print(f"  Downloading from {url[:80]}...")
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=120) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(1024 * 256)  # 256KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
                    else:
                        print(f"\r  {downloaded / 1e6:.1f} MB downloaded", end="", flush=True)
            print()
    except (HTTPError, URLError) as exc:
        print(f"\n  ERROR downloading {url}: {exc}")
        raise


def decompress_gzip(src: Path, dest_dir: Path) -> Path:
    """Decompress a .gz file and return the path to the decompressed file."""
    dest = dest_dir / src.name.replace(".gz", "")
    print(f"  Decompressing {src.name} -> {dest.name}")
    with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    src.unlink()
    return dest


def extract_from_zip(zip_path: Path, filenames: list[str], dest_dir: Path) -> list[Path]:
    """Extract specific files from a ZIP archive."""
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in filenames:
            matches = [n for n in zf.namelist() if n.endswith(name)]
            if not matches:
                print(f"  WARNING: {name} not found in {zip_path.name}")
                continue
            for match in matches:
                print(f"  Extracting {match} from {zip_path.name}")
                data = zf.read(match)
                dest = dest_dir / name
                dest.write_bytes(data)
                extracted.append(dest)
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets for Real Estate Tracker.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "raw"),
        help="Directory to save downloaded files (default: data/raw)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading datasets to: {output_dir}\n")

    success = 0
    failed = 0

    # --- Direct downloads ---
    for ds in DATASETS:
        name = ds["name"]
        filename = ds["filename"]
        dest = output_dir / filename

        print(f"[{name}]")

        if dest.exists() and dest.stat().st_size > 0:
            print(f"  Already exists ({dest.stat().st_size / 1e6:.1f} MB), skipping.\n")
            success += 1
            continue

        try:
            download_file(ds["url"], dest)

            if ds.get("decompress_gz"):
                decompress_gzip(dest, output_dir)

            if ds.get("extract_files"):
                extract_from_zip(dest, ds["extract_files"], output_dir)

            success += 1
            print(f"  Done.\n")
        except Exception as exc:
            print(f"  FAILED: {exc}\n")
            failed += 1

    print(f"\nFinished: {success} succeeded, {failed} failed.")

    # Print summary of files in output directory
    print(f"\nFiles in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name:50s} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()
