# SINCA internals

[SINCA](https://sinca.mma.gob.cl/) does not expose a documented public API. The
following documents how the library accesses its data, discovered via browser DevTools
inspection of the interactive map at `https://sinca.mma.gob.cl/mapainteractivo/index.html`.

## Backend architecture

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

## Station metadata discovery

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

## Parameter code discovery

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

## Data download endpoint

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

## Response format

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

## Station metadata (`sinca_stations.csv`)

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
