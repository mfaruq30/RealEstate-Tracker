"""Fetch data from external APIs and save to data/raw/.

Usage:
    python scripts/fetch_api_data.py [--output-dir data/raw]

    Environment variables (optional):
        CENSUS_API_KEY  — free key from https://api.census.gov/data/key_signup.html
        FRED_API_KEY    — free key from https://fred.stlouisfed.org/docs/api/fred/

APIs used:
    1. Census ACS 5-Year — median income, home values for MA ZIP codes
    2. FRED — 30-year fixed mortgage rate time series
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "RealEstateTracker/1.0 (BU CS506 Student Project)"
)

# Boston-area ZIP codes (Boston proper + key surrounding cities)
BOSTON_AREA_ZIPS = {
    # Boston proper
    "02101", "02102", "02103", "02104", "02105", "02106", "02107", "02108",
    "02109", "02110", "02111", "02112", "02113", "02114", "02115", "02116",
    "02117", "02118", "02119", "02120", "02121", "02122", "02123", "02124",
    "02125", "02126", "02127", "02128", "02129", "02130", "02131", "02132",
    "02133", "02134", "02135", "02136",
    # Cambridge
    "02138", "02139", "02140", "02141", "02142",
    # Somerville
    "02143", "02144", "02145",
    # Brookline
    "02445", "02446", "02447", "02467",
    # Newton
    "02458", "02459", "02460", "02461", "02462", "02464", "02465", "02466", "02468",
    # Quincy
    "02169", "02170", "02171",
    # Watertown / Waltham / Belmont
    "02452", "02453", "02454", "02472", "02478",
    # Chelsea / Revere / Everett
    "02149", "02150", "02151",
    # Medford / Malden
    "02148", "02155",
    # Arlington / Lexington
    "02420", "02421", "02474", "02476",
    # Dedham / Milton / Needham
    "02026", "02186", "02187", "02492", "02494",
}


def fetch_json(url: str) -> list | dict:
    """Fetch a URL and parse JSON response."""
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_census_acs(api_key: str | None, output_dir: Path) -> None:
    """Fetch Census ACS 5-Year data for MA ZIP codes.

    Variables:
        B25077_001E — Median home value (owner-occupied)
        B19013_001E — Median household income
        B01003_001E — Total population
        B25003_001E — Total housing units (tenure)
        B25003_002E — Owner-occupied units
        B25003_003E — Renter-occupied units
    """
    dest = output_dir / "census_acs_boston_zips.csv"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {dest.name} already exists, skipping.")
        return

    variables = "B25077_001E,B19013_001E,B01003_001E,B25003_001E,B25003_002E,B25003_003E"
    # Try 2022 first, fall back to 2021
    years = ["2022", "2021"]
    data = None

    for year in years:
        base_url = f"https://api.census.gov/data/{year}/acs/acs5"
        params = (
            f"?get=NAME,{variables}"
            f"&for=zip%20code%20tabulation%20area:*"
            f"&in=state:25"  # Massachusetts
        )
        if api_key:
            params += f"&key={api_key}"

        url = base_url + params
        print(f"  Fetching Census ACS {year} data for Massachusetts ZCTAs...")

        try:
            data = fetch_json(url)
            if data and len(data) >= 2:
                print(f"  Success with {year} ACS data.")
                break
        except (HTTPError, URLError) as exc:
            print(f"  {year} failed: {exc}, trying next year...")
            continue

    if data is None:
        print(f"  ERROR: All Census API attempts failed.")
        print(f"  TIP: Get a free API key at https://api.census.gov/data/key_signup.html")
        print(f"       Set CENSUS_API_KEY environment variable and re-run.")
        return

    if not data or len(data) < 2:
        print("  ERROR: No data returned from Census API.")
        return

    headers = data[0]
    rows = data[1:]

    # Map Census column names to friendlier names
    column_map = {
        "NAME": "zcta_name",
        "B25077_001E": "median_home_value",
        "B19013_001E": "median_household_income",
        "B01003_001E": "total_population",
        "B25003_001E": "total_housing_units",
        "B25003_002E": "owner_occupied_units",
        "B25003_003E": "renter_occupied_units",
        "state": "state_fips",
        "zip code tabulation area": "zcta",
    }

    renamed_headers = [column_map.get(h, h) for h in headers]

    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(renamed_headers)
        writer.writerows(rows)

    print(f"  Saved {len(rows)} ZCTAs to {dest.name}")

    # Count how many are in the Boston area
    zcta_idx = headers.index("zip code tabulation area")
    boston_count = sum(1 for row in rows if row[zcta_idx] in BOSTON_AREA_ZIPS)
    print(f"  ({boston_count} are in the Greater Boston area)")


def fetch_fred_mortgage_rates(api_key: str | None, output_dir: Path) -> None:
    """Fetch FRED 30-Year Fixed Rate Mortgage Average (MORTGAGE30US).

    Falls back to a direct CSV download if no API key is set.
    """
    dest = output_dir / "fred_mortgage_rates.csv"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {dest.name} already exists, skipping.")
        return

    if api_key:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id=MORTGAGE30US"
            f"&api_key={api_key}"
            f"&file_type=json"
            f"&observation_start=2000-01-01"
        )
        print("  Fetching FRED mortgage rate data via API...")
        try:
            data = fetch_json(url)
            observations = data.get("observations", [])

            with open(dest, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "mortgage_rate_30yr"])
                for obs in observations:
                    if obs["value"] != ".":
                        writer.writerow([obs["date"], obs["value"]])

            print(f"  Saved {len(observations)} observations to {dest.name}")
            return
        except (HTTPError, URLError) as exc:
            print(f"  API request failed ({exc}), trying CSV fallback...")

    # Fallback: direct CSV download from FRED (no API key needed)
    csv_url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        "?id=MORTGAGE30US"
        "&cosd=2000-01-01"
    )
    print("  Downloading FRED mortgage rate CSV (no API key)...")
    try:
        req = Request(csv_url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=120) as resp:
            content = resp.read().decode("utf-8")
            with open(dest, "w", encoding="utf-8") as f:
                f.write(content)
        line_count = content.count("\n")
        print(f"  Saved ~{line_count} rows to {dest.name}")
    except Exception as exc:
        print(f"  ERROR: Could not download FRED data: {exc}")
        print(f"  This is optional data — the pipeline will work without it.")
        print(f"  TIP: Get a free API key at https://fred.stlouisfed.org/docs/api/fred/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch API data for Real Estate Tracker.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "raw"),
        help="Directory to save fetched data (default: data/raw)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    census_key = os.environ.get("CENSUS_API_KEY")
    fred_key = os.environ.get("FRED_API_KEY")

    print(f"Fetching API data to: {output_dir}\n")

    if not census_key:
        print("NOTE: CENSUS_API_KEY not set. Census requests may be rate-limited.")
        print("      Get a free key: https://api.census.gov/data/key_signup.html\n")

    print("[Census ACS 5-Year Data]")
    fetch_census_acs(census_key, output_dir)
    print()

    print("[FRED 30-Year Mortgage Rate]")
    fetch_fred_mortgage_rates(fred_key, output_dir)
    print()

    print("API data fetch complete.")


if __name__ == "__main__":
    main()
