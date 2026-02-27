# Output columns & data lineage

Every DataFrame returned by `get_data()` / `get_data_async()` contains, alongside the measurement columns, companion **`dl.*` lineage columns** that record the status of every individual value. This lets you filter, audit, or visualize data quality without any additional processing.

## `ChileAirQuality`

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

## `ChileClimateData`

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
