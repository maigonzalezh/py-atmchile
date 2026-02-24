# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2024

### Added
- `ChileAirQuality`: download PM10, PM25, CO, SO2, NOX, NO2, NO, O3, temp, RH, ws, wd
  from SINCA monitoring stations, with sync and async APIs
- `ChileAirQuality.get_data_async()`: parallel downloads via httpx with configurable concurrency
- Data curation rules: NOX consistency, PM2.5 ≤ PM10, wind direction range, RH range
- Download lineage columns (`dl.*`) and optional SINCA validation status columns (`s.*`)
- `ChileClimateData`: download Temperatura, PuntoRocio, Humedad, Viento, PresionQFE, PresionQFF
  from DMC meteorological stations, with sync and async APIs
- Bundled SINCA station table (`sinca_stations.csv`, 122 stations, 10 columns)
- Python 3.9+ compatibility (`from __future__ import annotations`)
- Minimum dependency versions: numpy>=1.26.0, pandas>=2.2.0, requests>=2.28.0, httpx>=0.23.0

### Improvements over the original R package
- **Asynchronous concurrent downloads**: the R implementation downloads each station/parameter/year
  sequentially (`for` loops + blocking `read.csv()` / `download.file()`); this port uses
  `asyncio` + `httpx` to run multiple requests in parallel
- **Vectorized CSV processing**: data parsing and curation use pandas column-wise operations
  (`bfill`, `pd.to_numeric`, boolean masks) instead of element-wise iteration

### Notes
- Initial Python port of the [AtmChile R package](https://github.com/franciscoxaxo/AtmChile)
  by Francisco Menares et al. (Universidad de Chile, FONDECYT 1200674)
- Port assisted by Claude (Anthropic) AI
