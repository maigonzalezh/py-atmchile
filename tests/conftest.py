"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import pytest
import requests

# URLs probed to check endpoint availability before running integration tests.
SINCA_CGI_PROBE = (
    "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi"
    "?outtype=xcl&macro=./RM/D14/Cal/PM25//PM25.horario.horario.ic"
    "&from=230101&to=230101"
    "&path=/usr/airviro/data/CONAMA/&lang=esp&rsrc=&macropath="
)
SINCA_STATIONS_PROBE = "https://sinca.mma.gob.cl/index.php/json/listadomapa2k19/"
DMC_PROBE = (
    "https://climatologia.meteochile.gob.cl/application/datos/getDatosSaclim/"
    "330019_2023_Temperatura_"
)


@pytest.fixture
def skip_if_sinca_unavailable() -> None:
    """Skip if the SINCA CGI data endpoint is unreachable."""
    try:
        requests.get(SINCA_CGI_PROBE, timeout=15)
    except requests.RequestException:
        pytest.skip("SINCA CGI endpoint not reachable")


@pytest.fixture
def skip_if_sinca_stations_unavailable() -> None:
    """Skip if the SINCA stations JSON endpoint is unreachable."""
    try:
        requests.get(SINCA_STATIONS_PROBE, timeout=10)
    except requests.RequestException:
        pytest.skip("SINCA stations endpoint not reachable")


@pytest.fixture
def skip_if_dmc_unavailable() -> None:
    """Skip if the DMC climate data endpoint is unreachable."""
    try:
        requests.get(DMC_PROBE, timeout=15)
    except requests.RequestException:
        pytest.skip("DMC climate endpoint not reachable")
