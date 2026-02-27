"""Integration tests against live SINCA and DMC endpoints.

All tests are marked with @pytest.mark.integration and are excluded from the
default pytest run (configured via addopts in pyproject.toml). Run them with:

    uv run pytest -m integration -v
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd
import pytest

from atmchile import ChileClimateData
from atmchile.air_quality_data import ChileAirQuality

# ---------------------------------------------------------------------------
# Air quality — sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_air_quality_get_data(skip_if_sinca_unavailable: None) -> None:
    """Short 7-day window returns a non-empty DataFrame with a PM25 column."""
    aq = ChileAirQuality()
    result = aq.get_data(
        stations="RM/D14",
        parameters="PM25",
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 7),
    )
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "PM25" in result.columns


@pytest.mark.integration
def test_integration_air_quality_single_day(skip_if_sinca_unavailable: None) -> None:
    """start == end: get_data returns a DataFrame without raising."""
    aq = ChileAirQuality()
    dt = datetime(2023, 6, 15)
    result = aq.get_data(stations="RM/D14", parameters="PM25", start=dt, end=dt)
    assert isinstance(result, pd.DataFrame)


@pytest.mark.integration
def test_integration_air_quality_year_boundary(skip_if_sinca_unavailable: None) -> None:
    """Dec→Jan crossover: get_data returns a DataFrame without raising."""
    aq = ChileAirQuality()
    result = aq.get_data(
        stations="RM/D14",
        parameters="PM25",
        start=datetime(2022, 12, 28),
        end=datetime(2023, 1, 3),
    )
    assert isinstance(result, pd.DataFrame)


@pytest.mark.integration
def test_integration_air_quality_end_is_now(skip_if_sinca_unavailable: None) -> None:
    """end = datetime.now(): get_data returns a DataFrame without raising."""
    aq = ChileAirQuality()
    result = aq.get_data(
        stations="RM/D14",
        parameters="PM25",
        start=datetime(2024, 1, 1),
        end=datetime.now(),
    )
    assert isinstance(result, pd.DataFrame)


@pytest.mark.integration
def test_integration_air_quality_region_filter(skip_if_sinca_unavailable: None) -> None:
    """Region filter for RM returns a DataFrame without raising."""
    aq = ChileAirQuality()
    result = aq.get_data(
        stations="RM",
        parameters="PM25",
        region=True,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 2),
    )
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Air quality — async
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_air_quality_get_data_async_long_window(
    skip_if_sinca_unavailable: None,
) -> None:
    """Long 4-year async window returns a DataFrame without raising."""
    aq = ChileAirQuality()
    result = asyncio.run(
        aq.get_data_async(
            stations="RM/D14",
            parameters="PM25",
            start=datetime(2022, 1, 1),
            end=datetime(2026, 1, 1),
        )
    )
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Climate — sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_climate_get_data(skip_if_dmc_unavailable: None) -> None:
    """Short month window returns a non-empty DataFrame."""
    cd = ChileClimateData()
    result = cd.get_data(
        stations="330019",
        parameters="Temperatura",
        start=datetime(2023, 6, 1),
        end=datetime(2023, 6, 30),
    )
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


@pytest.mark.integration
def test_integration_climate_leap_year(skip_if_dmc_unavailable: None) -> None:
    """Feb 29 start date: get_data returns a DataFrame without raising."""
    cd = ChileClimateData()
    result = cd.get_data(
        stations="330019",
        parameters="Temperatura",
        start=datetime(2024, 2, 29),
        end=datetime(2024, 3, 1),
    )
    assert isinstance(result, pd.DataFrame)


@pytest.mark.integration
def test_integration_climate_end_is_today(skip_if_dmc_unavailable: None) -> None:
    """end = datetime.now(): get_data returns a DataFrame without raising."""
    cd = ChileClimateData()
    result = cd.get_data(
        stations="330019",
        parameters="Temperatura",
        start=datetime(2025, 1, 1),
        end=datetime.now(),
    )
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Climate — async
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_climate_get_data_async_long_window(
    skip_if_dmc_unavailable: None,
) -> None:
    """Long 4-year async window returns a DataFrame without raising."""
    cd = ChileClimateData()
    result = asyncio.run(
        cd.get_data_async(
            stations="330019",
            parameters="Temperatura",
            start=datetime(2022, 1, 1),
            end=datetime(2026, 1, 1),
        )
    )
    assert isinstance(result, pd.DataFrame)
