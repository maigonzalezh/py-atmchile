# atmchile

![CI](https://github.com/maigonzalezh/py-atmchile/actions/workflows/ci.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/atmchile)](https://pypi.org/project/atmchile/)
[![Python](https://img.shields.io/pypi/pyversions/atmchile)](https://pypi.org/project/atmchile/)
[![codecov](https://codecov.io/gh/maigonzalezh/py-atmchile/graph/badge.svg)](https://codecov.io/gh/maigonzalezh/py-atmchile)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maigonzalezh/py-atmchile/blob/main/examples/atmchile_usage_guide.ipynb)

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
    stations="RM/D14",
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

Every DataFrame includes `dl.*` lineage columns alongside measurement values, recording
the download and curation status of each individual data point. For the full column
schema, `dl.*` status values, curation rules, and validation tiers for both classes,
see [docs/output-columns.md](docs/output-columns.md).

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

# Run tests with coverage (minimum threshold 90%)
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

SINCA has no publicly documented API. This library accesses data through reverse-engineered
endpoints of the underlying Airviro platform. For details on backend architecture, station
metadata discovery, parameter code mapping, URL construction, and response format, see
[docs/sinca-internals.md](docs/sinca-internals.md).

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
