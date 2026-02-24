"""Utility scripts for the atmchile project."""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA_DIR: Path = Path(__file__).parent / "data"

SINCA_STATIONS_URL: str = "https://sinca.mma.gob.cl/index.php/json/listadomapa2k19/"

CSV_COLUMNS: list[str] = [
    "city",
    "station_code",
    "latitude",
    "longitude",
    "station_name",
    "region",
    "network",
    "region_index",
    "access_type",
    "operator",
]

# Maps JSON "region" strings (after .strip()) to roman numeral abbreviations.
# Includes aliases to handle endpoint inconsistencies (casing, missing prepositions,
# trailing newlines in some entries).
REGION_MAP: dict[str, str] = {
    "Región de Arica y Parinacota": "XV",
    "Región de Tarapacá": "I",
    "Región de Antofagasta": "II",
    "Región de Atacama": "III",
    "Región de Coquimbo": "IV",
    "Región de Valparaíso": "V",
    "Región Metropolitana de Santiago": "RM",
    "Región del Libertador General Bernardo O'Higgins": "VI",
    "Región del Maule": "VII",
    "Región del Ñuble": "XVI",
    "Región de Ñuble": "XVI",  # alias without "del"
    "Región del Biobío": "VIII",
    "Región de La Araucanía": "IX",
    "Región de Los Ríos": "XIV",
    "Región de los Ríos": "XIV",  # alias lowercase
    "Región de Los Lagos": "X",
    "Región de los Lagos": "X",  # alias lowercase
    "Región de Aysén del General Carlos Ibáñez del Campo": "XI",
    "Región Aysén del General Carlos Ibáñez del Campo": "XI",  # alias without "de"
    "Región de Magallanes y de la Antártica Chilena": "XII",
}


def _build_station_code(roman: str, key: str) -> str:
    # RM is already a complete prefix; all other regions take the "R" prefix (e.g. RII/201).
    prefix = roman if roman == "RM" else f"R{roman}"
    return f"{prefix}/{key}"


def refresh_stations() -> None:
    """Fetch the current station list from SINCA and overwrite the bundled CSV.

    Fetches JSON from the live listadomapa2k19 endpoint, transforms each station
    record into the 10-column CSV schema, and writes the result directly to
    src/atmchile/data/sinca_stations.csv. No intermediate file is saved to disk.

    Usage:
        uv run refresh-stations
    """
    # 1. Fetch
    print(f"Fetching stations from {SINCA_STATIONS_URL} ...")
    try:
        response = requests.get(SINCA_STATIONS_URL, timeout=30)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        print(f"ERROR: Could not connect to SINCA endpoint: {exc}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out after 30 seconds.")
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        print(f"ERROR: HTTP {exc.response.status_code}: {exc}")
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # 2. Parse JSON in memory (no file written to disk)
    try:
        stations_raw = response.json()
    except ValueError as exc:
        print(f"ERROR: Could not parse JSON response: {exc}")
        print(f"       Response text (first 200 chars): {response.text[:200]}")
        sys.exit(1)

    if not isinstance(stations_raw, list):
        print(f"ERROR: Expected a JSON array at the top level, got {type(stations_raw).__name__}.")
        sys.exit(1)

    # 3. Transform
    rows: list[dict] = []
    unknown_regions: list[str] = []

    for s in stations_raw:
        region_str = s.get("region", "").strip()
        roman = REGION_MAP.get(region_str)
        if roman is None:
            unknown_regions.append(region_str)
            roman = "UNKNOWN"

        rows.append(
            {
                "city": s.get("comuna", ""),
                "station_code": _build_station_code(roman, s.get("key", "")),
                "latitude": s.get("latitud", ""),
                "longitude": s.get("longitud", ""),
                "station_name": s.get("nombre", ""),
                "region": roman,
                "network": s.get("red") or "",  # "red" can be None for some stations
                "region_index": s.get("regionindex", ""),
                "access_type": s.get("calificacion", ""),
                "operator": s.get("empresa", ""),
            }
        )

    if unknown_regions:
        for r in sorted(set(unknown_regions)):
            print(f"WARNING: unrecognised region string → {repr(r)} (written as region='UNKNOWN')")

    # 4. Write CSV
    output_path = DATA_DIR / "sinca_stations.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # 5. Summary
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Done. {len(rows)} stations written to {output_path}")
    print(f"Timestamp: {ts}")


def test_cov() -> None:
    """Run tests with coverage and minimum threshold of 80%."""
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=atmchile",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-fail-under=80",
    ] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)
