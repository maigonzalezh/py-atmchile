# atmchile

Python library to download air quality and climate data from Chilean government monitoring networks.

## Description

**atmchile** provides programmatic access to two national Chilean environmental
monitoring systems: SINCA (air quality) and DMC (climate). Both modules return
tidy pandas DataFrames annotated with download-lineage columns, and support
synchronous and asynchronous bulk downloads. Compared to the original R
implementation — which uses sequential `for` loops and processes each CSV
row-by-row — this port adds two performance improvements: **asynchronous
concurrent downloads** (`asyncio` + `httpx`) and **vectorized CSV processing**
via pandas (column-wise operations instead of element-wise iteration).

## Data Sources

### SINCA — air quality

[SINCA](https://sinca.mma.gob.cl/) (Sistema de Información Nacional de Calidad
del Aire) is Chile's national air quality monitoring network, operated by the
Ministry of Environment (MMA). It covers 120+ stations distributed across the
country's administrative regions, measuring hourly concentrations of criteria
pollutants and meteorological variables.

#### SINCA internals

[SINCA (Sistema de Información Nacional de Calidad del Aire)](https://sinca.mma.gob.cl/)
is Chile's national air quality monitoring system, managed by the Ministry of
Environment (MMA).

##### Backend architecture

SINCA runs on **Airviro**, a commercial air quality monitoring platform developed by
IVL Swedish Environmental Research Institute (Sweden). The backend is a CGI-based
system running on Apache/Linux, with data stored in a structured filesystem:

```
/usr/airviro/data/CONAMA/
└── {REGION}/{STATION_KEY}/
    ├── Cal/    # Calibrated pollutant measurements
    └── Met/    # Meteorological measurements
```

##### Station metadata discovery

SINCA does not expose a documented public API. The station list is loaded dynamically
by the interactive map at `https://sinca.mma.gob.cl/mapainteractivo/index.html`.
Inspecting the network traffic in browser DevTools while loading that page reveals the
JavaScript bundle `mapa.js`, which hardcodes the station list endpoint:

```javascript
// mapa.js
var g = { listado: "//sinca.mma.gob.cl/index.php/json/listadomapa2k19/" };
```

This endpoint returns a JSON array where each object represents a station:

```json
{
  "nombre":      "Alto Hospicio",
  "key":         "117",
  "latitud":     -20.290466881209,
  "longitud":    -70.100192427636,
  "comuna":      "Alto Hospicio",
  "red":         "Red MMA",
  "region":      "Región de Tarapacá",
  "regionindex": 2,
  "calificacion": "Pública",
  "empresa":     "Ministerio del Medio Ambiente",
  "realtime":    [...]
}
```

The `realtime` array contains the last 24 hours of readings and is not used for
station metadata. The fields retained in `sinca_stations.csv` are:

| JSON field | CSV column | Description |
|---|---|---|
| `nombre` | `station_name` | Human-readable station name |
| `key` | `station_code` | Unique station identifier used in download URLs |
| `latitud` | `latitude` | WGS84 latitude |
| `longitud` | `longitude` | WGS84 longitude |
| `comuna` | `city` | Municipality |
| `region` | `region` | Administrative region (roman numeral: `I`, `II`, ... `RM`) |
| `red` | `network` | Monitoring network name |
| `regionindex` | `region_index` | Numeric region index |
| `calificacion` | `access_type` | Access classification (`Pública` / `Privada`) |
| `empresa` | `operator` | Operating organization |

##### Parameter code discovery

The same `mapa.js` reveals the parameter codes used throughout SINCA, hardcoded in a
`switch` statement that maps internal codes to display labels:

```javascript
switch (e.realtime[T].code) {
  case "PM10":  B = "MP 10";                break;
  case "PM25":  B = "MP 2.5";              break;
  case "0001":  B = "Dióxido de azufre";   break;
  case "0003":  B = "Dióxido de nitrógeno"; break;
  case "0004":  B = "Monóxido de carbono"; break;
  case "0008":  B = "Ozono";               break;
}
```

These codes map directly to filesystem paths in the download endpoint. Meteorological
parameters (`temp`, `RH`, `ws`, `wd`) do not appear in `mapa.js` — their paths were
discovered empirically by probing the CGI endpoint.

##### Data download endpoint

All historical data is served by a single public CGI endpoint with no authentication
required. The URL is constructed as:

```
https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi
  ?outtype=xcl
  &macro=./{STATION_CODE}{PARAMETER_PATH}from={YYMMDD}&to={YYMMDD}
  &path=/usr/airviro/data/CONAMA/
  &lang=esp&rsrc=&macropath=
```

The full set of parameter paths, as used by this library:

| Parameter | Type | Path |
|---|---|---|
| `PM10` | Pollutant | `/Cal/PM10//PM10.horario.horario.ic&` |
| `PM25` | Pollutant | `/Cal/PM25//PM25.horario.horario.ic&` |
| `SO2` | Pollutant | `/Cal/0001//0001.horario.horario.ic&` |
| `NO` | Pollutant | `/Cal/0002//0002.horario.horario.ic&` |
| `NO2` | Pollutant | `/Cal/0003//0003.horario.horario.ic&` |
| `CO` | Pollutant | `/Cal/0004//0004.horario.horario.ic&` |
| `O3` | Pollutant | `/Cal/0008//0008.horario.horario.ic&` |
| `NOX` | Pollutant | `/Cal/0NOX//0NOX.horario.horario.ic&` |
| `temp` | Meteorological | `/Met/TEMP//horario_000.ic&` (alt: `horario_010.ic&`) |
| `RH` | Meteorological | `/Met/RHUM//horario_000.ic&` (alt: `horario_002.ic&`) |
| `ws` | Meteorological | `/Met/WSPD//horario_000.ic&` (alt: `horario_010.ic&`) |
| `wd` | Meteorological | `/Met/WDIR//horario_000_spec.ic&` (alt: `horario_010_spec.ic&`) |

Meteorological parameters have an alternative path tried automatically when the
primary returns no data. The date parameters use 6-digit format: `YYMMDD` (e.g.,
`200101` for January 1st, 2020).

**Example** — PM10 from Parque O'Higgins (`RM/D14`), January 2020:

```
https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?outtype=xcl&macro=./RM/D14/Cal/PM10//PM10.horario.horario.ic&from=200101&to=200131&path=/usr/airviro/data/CONAMA/&lang=esp&rsrc=&macropath=
```

##### Response format

The endpoint returns a semicolon-delimited CSV with hourly records and three
validation columns in priority order:

| Column | Description |
|---|---|
| `FECHA (YYMMDD)` | Date in `YYMMDD` format |
| `HORA (HHMM)` | Time in `HHMM` format |
| `Registros validados` | Operationally validated values (highest priority) |
| `Registros preliminares` | Preliminary values |
| `Registros no validados` | Raw, unvalidated values (lowest priority) |

The library coalesces these three columns left-to-right, taking the first non-null
value per row.

##### Station metadata (`sinca_stations.csv`)

The file `src/atmchile/data/sinca_stations.csv` is the **source of truth** for SINCA
station metadata used by this library. It contains 118 stations across 22 monitoring
networks, with columns: `city`, `station_code`, `latitude`, `longitude`,
`station_name`, `region`, `network`, `region_index`, `access_type`, `operator`.

This file is kept up to date by running:

```bash
uv run refresh-stations
```

That command fetches the live `listadomapa2k19` endpoint, transforms the JSON in
memory, and overwrites the CSV — no intermediate file is written to disk. See
[Refresh station metadata](#refresh-station-metadata) under Development for details.

> **Note:** The endpoint name (`listadomapa2k19`) and the parameter codes were
> discovered via source inspection and are not officially documented by MMA. Both
> are subject to change in future SINCA updates.

### DMC — climate

[DMC](https://climatologia.meteochile.gob.cl/) (Dirección Meteorológica de
Chile) is Chile's national weather service, operated by the Dirección General
de Aeronáutica Civil (DGAC). It provides synoptic climate observations
(temperature, dew point, humidity, wind, surface and sea-level pressure) from
stations nationwide.

## Available Parameters

### ChileAirQuality (SINCA)

| Parameter | Description | Unit |
|---|---|---|
| `PM10` | Particulate matter ≤ 10 µm (hourly average) | µg/m³N |
| `PM25` | Particulate matter ≤ 2.5 µm (hourly average) | µg/m³N |
| `SO2` | Sulfur dioxide | µg/m³N |
| `NOX` | Nitrogen oxides (NO + NO₂) | ppb |
| `NO2` | Nitrogen dioxide | ppb |
| `NO` | Nitric oxide | ppb |
| `O3` | Tropospheric ozone | ppb |
| `CO` | Carbon monoxide | ppb |
| `temp` | Air temperature | °C |
| `RH` | Relative humidity | % |
| `ws` | Wind speed | m/s |
| `wd` | Wind direction (meteorological convention) | ° |

### ChileClimateData (DMC)

| Parameter | Description | Output column(s) | Unit |
|---|---|---|---|
| `Temperatura` | Air temperature | `Ts` | °C |
| `PuntoRocio` | Dew point temperature | `Td` | °C |
| `Humedad` | Relative humidity | `HR` | % |
| `Viento` | Wind | `dd` (direction), `ff` (speed), `VRB` (variability flag) | ° / m/s / — |
| `PresionQFE` | Station-level pressure (QFE) | `QFE` | hPa |
| `PresionQFF` | Sea-level pressure (QFF) | `QFF` | hPa |

## Installation

### Requirements

- Python >= 3.9
- [uv](https://github.com/astral-sh/uv)

### Setup

#### 1. Install uv

If you don't have `uv` installed:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

#### 2. Clone the repository

```bash
git clone <repository-url>
cd atmchile
```

#### 3. Set up virtual environment and install dependencies

```bash
# Install all dependencies (including development dependencies)
uv sync --all-extras

# Or only development dependencies
uv sync --extra dev
```

This will create a virtual environment in `.venv/` and install:
- **Main dependencies**: numpy, pandas, requests, httpx
- **Development dependencies**: ruff, pytest, pytest-asyncio, pytest-cov

#### 4. Verify installation

```bash
# Verify that the environment is active and dependencies are installed
uv run python -c "from atmchile import ChileClimateData; print('✓ Installation successful')"
```

## Quick Start

```python
from datetime import datetime
from atmchile import ChileAirQuality

caq = ChileAirQuality()
df = caq.get_data(
    stations="El Bosque",
    parameters=["PM10", "PM25"],
    start=datetime(2020, 1, 1),
    end=datetime(2020, 1, 31),
)
print(df.head())
```

## Usage

### ChileAirQuality

```python
from datetime import datetime
from atmchile import ChileAirQuality

# Create instance
caq = ChileAirQuality()

# Get air quality data
data = caq.get_data(
    stations="El Bosque",
    parameters=["PM10", "PM25"],
    start=datetime(2020, 1, 1),
    end=datetime(2020, 1, 2),
)

print(data)
```

```python
import asyncio
from datetime import datetime
from atmchile import ChileAirQuality

async def main():
    caq = ChileAirQuality(max_concurrent_downloads=5)

    data = await caq.get_data_async(
        stations=["RM", "II"],
        parameters=["PM10", "PM25"],
        start=datetime(2020, 1, 1),
        end=datetime(2021, 12, 31),
        region=True,
    )

    print(data)

asyncio.run(main())
```

### ChileClimateData

```python
from datetime import datetime
from atmchile import ChileClimateData

# Create instance
ccd = ChileClimateData()

# Get data synchronously
data = ccd.get_data(
    stations="180005",
    parameters=["Temperatura", "Humedad"],
    start=datetime(2020, 1, 1, 0, 0, 0),
    end=datetime(2020, 12, 31, 23, 0, 0),
)

print(data)
```

```python
import asyncio
from datetime import datetime
from atmchile import ChileClimateData

async def main():
    ccd = ChileClimateData(max_concurrent_downloads=5)

    data = await ccd.get_data_async(
        stations=["RM", "II"],
        parameters=["Temperatura", "PresionQFE"],
        start=datetime(2020, 1, 1, 0, 0, 0),
        end=datetime(2021, 12, 31, 23, 0, 0),
        region=True,
    )

    print(data)

asyncio.run(main())
```

## Output columns & data lineage

Every DataFrame returned by `get_data()` / `get_data_async()` contains, alongside the measurement columns, companion **`dl.*` lineage columns** that record the status of every individual value. This lets you filter, audit, or visualize data quality without any additional processing.

### `ChileAirQuality`

For each requested parameter `{param}`, the output includes:

| Column | Always present | Description |
|---|---|---|
| `{param}` | yes | Measurement value, or `NaN` if missing or curated |
| `dl.{param}` | yes | Download / curation status (see below) |
| `s.{param}` | only if `st=True` | SINCA source validation tier (see below) |

**`dl.{param}` — possible values:**

| Value | Meaning |
|---|---|
| `ok` | Value was downloaded and is present |
| `empty` | Row exists in source but no measurement was reported for that hour |
| `download_error` | HTTP request failed or response could not be parsed; all rows for this parameter carry this status |
| `curated` | Value was present (`ok`) but a curation rule flagged it as physically inconsistent; measurement set to `NaN` |

**`s.{param}` — SINCA validation tier (only when `st=True`):**

| Value | SINCA source column | Meaning |
|---|---|---|
| `V` | Registros validados | Operationally validated by the monitoring network |
| `PV` | Registros preliminares | Preliminary, subject to revision |
| `NV` | Registros no validados | Raw, unvalidated |
| `""` | — | No value in any validation column for that row |

**Curation rules** (applied when `curate=True`, which is the default):

| Rule | Condition | Parameters nullified |
|---|---|---|
| NOX consistency | NO + NO₂ > NOX × 1.001 | `NO`, `NO2`, `NOX` |
| PM consistency | PM2.5 > PM10 × 1.001 | `PM10`, `PM25` |
| Wind direction range | `wd` < 0 or `wd` > 360 | `wd` |
| Relative humidity range | `RH` < 0 or `RH` > 100 | `RH` |

### `ChileClimateData`

For each output column `{col}` (e.g. `Ts`, `HR`, `dd`, `ff`):

| Column | Always present | Description |
|---|---|---|
| `{col}` | yes | Measurement value, or `NaN` if missing |
| `dl.{col}` | yes | Download status (see below) |

**`dl.{col}` — possible values:**

| Value | Meaning |
|---|---|
| `ok` | Value is present (non-NaN) in the downloaded data |
| `empty` | Row exists but the value is `NaN` in the source |
| `download_error` | HTTP request failed or response could not be parsed; all rows for this column carry this status |

> `ChileClimateData` does not apply curation rules — there is no `curated` status for this class.

## Development

### Run tests

```bash
# Run all tests
uv run pytest

# Run with more verbosity
uv run pytest -v

# Run a specific test
uv run pytest tests/test_climate_data.py::test_init_with_default_path

# Run tests from a specific file
uv run pytest tests/test_climate_data.py

# Run tests with coverage (minimum threshold 80%)
uv run test-cov
```

#### Test coverage

- **`uv run test-cov`**: Runs tests with coverage and generates reports:
  - Terminal report with uncovered lines
  - HTML report in `htmlcov/index.html`
  - **Minimum threshold**: 80% (tests will fail if coverage is lower)

```bash
# macOS
open htmlcov/index.html

# Linux
xdg-open htmlcov/index.html
```

### Refresh station metadata

```bash
uv run refresh-stations
```

Fetches the current station list from the SINCA `listadomapa2k19` endpoint and
overwrites `src/atmchile/data/sinca_stations.csv`. Run this whenever SINCA adds or
removes stations. The command prints a summary of stations written and a UTC timestamp.

If any station has an unrecognised region string, a `WARNING` line is printed and
the station is written with `region='UNKNOWN'` so data loss is visible rather than
silent. In that case, update `REGION_MAP` in `src/atmchile/scripts.py` before
committing the refreshed CSV.

### Linting and formatting

```bash
# Check code (linting)
uv run ruff check .

# Format code
uv run ruff format .

# Check and format in one step
uv run ruff check . --fix
uv run ruff format .
```

### Update dependencies

```bash
# Update all dependencies to the latest compatible versions
uv lock --upgrade

# Sync environment with new versions
uv sync --all-extras
```

### Install in development mode

```bash
# The project is already in editable mode with uv sync
# If you need to reinstall:
uv pip install -e .
```

### Project Structure

```
atmchile/
├── src/atmchile/
│   ├── __init__.py
│   ├── climate_data.py        # Main ChileClimateData class
│   ├── air_quality_data.py    # Main ChileAirQuality class
│   ├── scripts.py             # Utility scripts
│   ├── utils.py               # Utilities
│   └── data/
│       ├── dmc_stations.csv   # Meteorological stations table
│       └── sinca_stations.csv # SINCA air quality stations table
├── tests/
│   ├── test_air_quality.py       # Tests for ChileAirQuality
│   ├── test_air_quality_async.py # Tests for async ChileAirQuality
│   ├── test_climate_data.py      # Tests for ChileClimateData
│   └── test_utils.py             # Tests for utilities
├── example_usage.py          # Usage examples
├── pyproject.toml            # Project configuration and dependencies
├── uv.lock                   # Dependency lockfile (generated by uv)
├── README.md
└── LICENSE
```

### Dependencies

#### Main dependencies (runtime)

- `numpy>=1.26.0` - Numerical operations
- `pandas>=2.2.0` - Data manipulation
- `requests>=2.28.0` - Synchronous HTTP client
- `httpx>=0.23.0` - Asynchronous HTTP client

#### Development dependencies

- `ruff>=0.14.0` - Linter and formatter
- `pytest>=7.0.0` - Testing framework
- `pytest-asyncio>=0.23.0` - Support for async tests
- `pytest-cov>=4.1.0` - pytest plugin for measuring code coverage

## Attribution & AI Disclosure

This library is a Python port of the [AtmChile R package](https://github.com/franciscoxaxo/AtmChile)
by Francisco Menares et al., developed at the Universidad de Chile under FONDECYT Project 1200674.

The initial R-to-Python port was generated with the assistance of **Claude (Anthropic) AI**.
Subsequent architecture decisions, testing, and incremental improvements were made by the author.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE),
inherited from the original AtmChile R package.
