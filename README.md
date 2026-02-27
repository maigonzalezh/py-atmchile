# atmchile

![CI](https://github.com/maigonzalezh/py-atmchile/actions/workflows/ci.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/atmchile)](https://pypi.org/project/atmchile/)
[![Python](https://img.shields.io/pypi/pyversions/atmchile)](https://pypi.org/project/atmchile/)

Python library to download air quality and climate data from two Chilean government monitoring
networks: **SINCA** (Sistema de Información Nacional de Calidad del Aire), operated by the
Ministry of Environment of Chile (MMA — Ministerio del Medio Ambiente), and **DMC**
(Dirección Meteorológica de Chile). Both modules return tidy pandas DataFrames with
download-lineage columns, and support synchronous and asynchronous bulk downloads.

## Installation

```bash
pip install atmchile
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

> Units: **µg/m³N** = micrograms per normal cubic metre (0 °C, 1 atm); **ppb** = parts per billion by volume.

### ChileClimateData (DMC)

| Parameter | Description | Output column(s) | Unit |
|---|---|---|---|
| `Temperatura` | Air temperature | `Ts` | °C |
| `PuntoRocio` | Dew point temperature | `Td` | °C |
| `Humedad` | Relative humidity | `HR` | % |
| `Viento` | Wind | `dd` (direction), `ff` (speed), `VRB` (variability flag) | ° / m/s / — |
| `PresionQFE` | Station-level pressure (QFE) | `QFE` | hPa |
| `PresionQFF` | Sea-level pressure (QFF) | `QFF` | hPa |

## Usage

### ChileAirQuality

```python
from datetime import datetime
from atmchile import ChileAirQuality

caq = ChileAirQuality()

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

ccd = ChileClimateData()

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

## Data Sources

### SINCA — air quality

[SINCA](https://sinca.mma.gob.cl/) (Sistema de Información Nacional de Calidad del Aire) is Chile's national air quality monitoring network, operated by the Ministry of Environment (MMA). It covers 120+ stations distributed across the country's administrative regions, measuring hourly concentrations of criteria pollutants and meteorological variables.

### DMC — climate

[DMC](https://climatologia.meteochile.gob.cl/) (Dirección Meteorológica de Chile) is Chile's national weather service, operated by the Dirección General de Aeronáutica Civil (DGAC). It provides synoptic climate observations (temperature, dew point, humidity, wind, surface and sea-level pressure) from stations nationwide.

## Development

### Run tests

```bash
# Run all tests
uv run pytest

# Run with more verbosity
uv run pytest -v

# Run tests with coverage (minimum threshold 80%)
uv run test-cov
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
uv run ruff check . --fix
uv run ruff format .
```

### Update dependencies

```bash
uv lock --upgrade && uv sync --all-extras
```

## SINCA internals

[SINCA](https://sinca.mma.gob.cl/) does not expose a documented public API. The
following documents how the library accesses its data, discovered via browser DevTools
inspection of the interactive map at `https://sinca.mma.gob.cl/mapainteractivo/index.html`.

### Backend architecture

SINCA runs on **Airviro**, a commercial air quality monitoring platform developed by
IVL Swedish Environmental Research Institute (Sweden). The backend is a CGI (Common
Gateway Interface) system running on Apache/Linux, with data stored in a structured
filesystem:

```
/usr/airviro/data/CONAMA/
└── {REGION}/{STATION_KEY}/
    ├── Cal/    # Calibrated pollutant measurements
    └── Met/    # Meteorological measurements
```

### Station metadata discovery

The station list is loaded dynamically by the interactive map. Inspecting the network
traffic reveals the JavaScript bundle `mapa.js`, which hardcodes the station list endpoint:

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
| `latitud` | `latitude` | WGS84 (World Geodetic System 1984) latitude |
| `longitud` | `longitude` | WGS84 longitude |
| `comuna` | `city` | Municipality |
| `region` | `region` | Administrative region (roman numeral: `I`, `II`, ... `RM`) |
| `red` | `network` | Monitoring network name |
| `regionindex` | `region_index` | Numeric region index |
| `calificacion` | `access_type` | Access classification (`Pública` / `Privada`) |
| `empresa` | `operator` | Operating organization |

### Parameter code discovery

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

### Data download endpoint

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

### Response format

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

### Station metadata (`sinca_stations.csv`)

The file `src/atmchile/data/sinca_stations.csv` is the **source of truth** for SINCA
station metadata used by this library. It contains 118 stations across 22 monitoring
networks, with columns: `city`, `station_code`, `latitude`, `longitude`,
`station_name`, `region`, `network`, `region_index`, `access_type`, `operator`.

This file is kept up to date by running:

```bash
uv run refresh-stations
```

> **Note:** The endpoint name (`listadomapa2k19`) and the parameter codes were
> discovered via source inspection and are not officially documented by MMA. Both
> are subject to change in future SINCA updates.

## Attribution & AI Disclosure

This library is a Python port of the [AtmChile R package](https://github.com/franciscoxaxo/AtmChile)
by Francisco Menares et al., developed at the Department of Chemistry, Faculty of Sciences,
Universidad de Chile. The original work was funded by ANID/FONDECYT Grant No. 1200674
(Fondo Nacional de Desarrollo Científico y Tecnológico, Chile's national research funding agency).

The original package is described in:

> Menares et al. (2022). [The AtmChile Open-Source Interactive Application for Exploring Air Quality and Meteorological Data in Chile](https://www.mdpi.com/2073-4433/13/9/1364). *Atmosphere*, 13(9), 1364.

The initial R-to-Python port was generated with the assistance of **Claude (Anthropic) AI**.
Subsequent architecture decisions, testing, and incremental improvements were made by the author.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE),
inherited from the original AtmChile R package.
