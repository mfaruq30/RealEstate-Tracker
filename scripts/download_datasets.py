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
        "name": "Boston Parcels 2025 (CSV)",
        "url": (
            "https://data.boston.gov/dataset/"
            "d0fe512c-80bd-4ed7-8582-c6ec38aedfa0/resource/"
            "01c6e34d-56c2-4442-958c-518951f4af71/download/"
            "parcels_2025.csv"
        ),
        "filename": "boston_parcels_2025.csv",
    },
    {
        "name": "Zillow ZHVI by ZIP Code",
        "url": (
            "https://files.zillowstatic.com/research/public_csvs/zhvi/"
            "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
        ),
        "filename": "zillow_zhvi_by_zip.csv",
    },
    {
        "name": "Redfin ZIP-Level Market Tracker",
        "url": (
            "https://redfin-public-data.s3.us-west-2.amazonaws.com/"
            "redfin_market_tracker/zip_code_market_tracker.tsv000.gz"
        ),
        "filename": "redfin_zip_market_tracker.tsv.gz",
        "decompress_gz": True,
    },
    {
        "name": "MBTA GTFS Feed",
        "url": "https://cdn.mbta.com/MBTA_GTFS.zip",
        "filename": "mbta_gtfs.zip",
        "extract_files": ["stops.txt"],
    },
    {
        "name": "BPD Crime Incident Reports 2024",
        "url": (
            "https://data.boston.gov/dataset/"
            "crime-incident-reports-august-2015-to-date-source-new-system/"
            "resource/313e56df-6d77-49d2-9c49-ee20f0c46c98/download/"
            "tmppdf54_k_.csv"
        ),
        "filename": "bpd_crime_incidents_2024.csv",
    },
]

# Kaggle datasets — require `kaggle` CLI (pip install kaggle) and API token.
# See: https://www.kaggle.com/docs/api
KAGGLE_DATASETS: list[dict] = [
    {
        "name": "USA Real Estate Dataset (200K+ with descriptions)",
        "slug": "ahmedshahriarsakib/usa-real-estate-dataset",
        "filename": "usa_real_estate_listings.csv",
    },
    {
        "name": "Ames Housing Dataset",
        "competition": "house-prices-advanced-regression-techniques",
        "filename": "train.csv",
        "rename_to": "ames_housing.csv",
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


def download_kaggle_datasets(output_dir: Path) -> tuple[int, int]:
    """Download datasets from Kaggle. Requires `kaggle` CLI and API token.

    Returns (success_count, failed_count).
    """
    try:
        import subprocess
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\n[Kaggle Datasets]")
        print("  kaggle CLI not found. Install with: pip install kaggle")
        print("  Then set up API token: https://www.kaggle.com/docs/api")
        print("  Skipping Kaggle datasets.\n")
        return 0, 0

    import subprocess
    success = 0
    failed = 0

    for ds in KAGGLE_DATASETS:
        name = ds["name"]
        final_name = ds.get("rename_to", ds["filename"])
        dest = output_dir / final_name

        print(f"[Kaggle: {name}]")

        if dest.exists() and dest.stat().st_size > 0:
            print(f"  Already exists ({dest.stat().st_size / 1e6:.1f} MB), skipping.\n")
            success += 1
            continue

        try:
            if "competition" in ds:
                cmd = [
                    "kaggle", "competitions", "download",
                    "-c", ds["competition"],
                    "-f", ds["filename"],
                    "-p", str(output_dir),
                ]
            else:
                cmd = [
                    "kaggle", "datasets", "download",
                    "-d", ds["slug"],
                    "-p", str(output_dir),
                    "--unzip",
                ]

            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"  ERROR: {result.stderr.strip()}")
                failed += 1
                continue

            # Handle rename if needed
            if "rename_to" in ds:
                src = output_dir / ds["filename"]
                if src.exists():
                    src.rename(dest)
                    print(f"  Renamed {ds['filename']} -> {final_name}")

            # Handle zip from competition downloads
            zip_dest = output_dir / f"{ds['filename']}.zip"
            if zip_dest.exists():
                extract_from_zip(zip_dest, [ds["filename"]], output_dir)
                if "rename_to" in ds:
                    src = output_dir / ds["filename"]
                    if src.exists():
                        src.rename(dest)

            success += 1
            print(f"  Done.\n")
        except Exception as exc:
            print(f"  FAILED: {exc}\n")
            failed += 1

    return success, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets for Real Estate Tracker.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "raw"),
        help="Directory to save downloaded files (default: data/raw)",
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Skip Kaggle dataset downloads",
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

    # --- Kaggle datasets ---
    if not args.skip_kaggle:
        kg_success, kg_failed = download_kaggle_datasets(output_dir)
        success += kg_success
        failed += kg_failed

    print(f"\nFinished: {success} succeeded, {failed} failed.")

    # Print summary of files in output directory
    print(f"\nFiles in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name:50s} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()
