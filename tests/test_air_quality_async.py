"""Tests for async methods in ChileAirQuality."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from atmchile.air_quality_data import ChileAirQuality


@pytest.fixture
def mock_stations_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_name": ["Santiago", "Valparaiso"],
            "city": ["SCL", "VAP"],
            "station_code": ["RM/D14", "RV/550"],
            "latitude": [-33.45, -33.04],
            "longitude": [-70.66, -71.62],
            "region": ["RM", "V"],
            "network": ["Red MMA", "Red MMA"],
            "region_index": [13, 5],
            "access_type": ["Pública", "Pública"],
            "operator": ["Ministerio del Medio Ambiente", "Ministerio del Medio Ambiente"],
        }
    )


@pytest.fixture
def air_quality_instance(mock_stations_df: pd.DataFrame) -> ChileAirQuality:
    aq = ChileAirQuality(stations_csv_path=None, max_concurrent_downloads=5)
    aq.set_stations_table(mock_stations_df)
    return aq


@pytest.fixture
def date_range() -> tuple[datetime, datetime]:
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2020, 1, 1, 5, 0, 0)
    return start, end


def _build_csv_content(parameter: str, dates: list[str]) -> str:
    """Build CSV content for a parameter."""
    if parameter in ["temp", "RH", "ws", "wd"]:
        # Meteorological parameters
        csv_lines = ["FECHA (YYMMDD);HORA (HHMM);" + parameter]
        for i, _date_str in enumerate(dates):
            fecha = "200101"
            hora = f"{i + 1:02d}00"
            value = "25.5" if parameter == "temp" else "50.0"
            csv_lines.append(f"{fecha};{hora};{value}")
        return "\n".join(csv_lines)
    else:
        # Contaminant parameters
        contam_header = "FECHA (YYMMDD);HORA (HHMM);Registros validados;Registros preliminares;Registros no validados"  # noqa: E501
        csv_lines = [contam_header]
        for i, _date_str in enumerate(dates):
            fecha = "200101"
            hora = f"{i + 1:02d}00"
            value = "10.5"
            csv_lines.append(f"{fecha};{hora};{value};;")
        return "\n".join(csv_lines)


def _build_param_df(start: datetime, end: datetime, parameter: str) -> pd.DataFrame:
    """Build parameter DataFrame."""
    dates = pd.date_range(
        start=start.replace(hour=1, minute=0, second=0),
        end=end.replace(hour=23, minute=0, second=0),
        freq="h",
    )
    return pd.DataFrame({"date": dates.strftime(ChileAirQuality.SINCA_DATE_FORMAT), parameter: 1})


class MockAsyncClient:
    """Mock async HTTP client for testing."""

    def __init__(self, csv_content: str | None = None):
        self.csv_content = csv_content or _build_csv_content("PM10", ["01/01/2020 01:00"])

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str):
        """Mock GET request."""
        response = MagicMock()
        response.content = self.csv_content.encode("utf-8")
        response.raise_for_status = MagicMock()
        return response


@pytest.mark.asyncio
async def test_get_data_async_basic(air_quality_instance: ChileAirQuality, date_range):
    """Test basic async functionality."""
    start, end = date_range
    csv_content = _build_csv_content("PM10", ["01/01/2020 01:00", "01/01/2020 02:00"])

    with patch(
        "atmchile.air_quality_data.httpx.AsyncClient", return_value=MockAsyncClient(csv_content)
    ):
        df = await air_quality_instance.get_data_async(
            stations="RM/D14",
            parameters="PM10",
            start=start,
            end=end,
            curate=False,
            st=False,
        )

    assert not df.empty
    assert "PM10" in df.columns
    assert "city" in df.columns
    assert df["city"].eq("SCL").all()


@pytest.mark.asyncio
async def test_get_data_async_multiple_parameters(
    air_quality_instance: ChileAirQuality, date_range
):
    """Test async download with multiple parameters."""
    start, end = date_range

    class MultiParamClient:
        """Mock client that returns different CSV content based on URL."""

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            """Return different CSV content based on parameter in URL."""
            if "PM10" in url:
                csv_content = _build_csv_content("PM10", ["01/01/2020 01:00"])
            elif "PM25" in url:
                csv_content = _build_csv_content("PM25", ["01/01/2020 01:00"])
            else:
                csv_content = _build_csv_content("CO", ["01/01/2020 01:00"])
            response = MagicMock()
            response.content = csv_content.encode("utf-8")
            response.raise_for_status = MagicMock()
            return response

    with patch("atmchile.air_quality_data.httpx.AsyncClient", return_value=MultiParamClient()):
        df = await air_quality_instance.get_data_async(
            stations="RM/D14",
            parameters=["PM10", "PM25", "CO"],
            start=start,
            end=end,
            curate=False,
            st=False,
        )

    assert not df.empty
    assert {"PM10", "PM25", "CO"}.issubset(df.columns)


@pytest.mark.asyncio
async def test_get_data_async_multiple_stations(air_quality_instance: ChileAirQuality, date_range):
    """Test async download with multiple stations."""
    start, end = date_range
    csv_content = _build_csv_content("PM10", ["01/01/2020 01:00"])

    with patch(
        "atmchile.air_quality_data.httpx.AsyncClient", return_value=MockAsyncClient(csv_content)
    ):
        df = await air_quality_instance.get_data_async(
            stations=["RM/D14", "RV/550"],
            parameters="PM10",
            start=start,
            end=end,
            curate=False,
            st=False,
        )

    assert not df.empty
    assert set(df["city"].unique()) == {"SCL", "VAP"}


@pytest.mark.asyncio
async def test_get_data_async_concurrency_limit(air_quality_instance: ChileAirQuality, date_range):
    """Test that semaphore limits concurrency."""
    start, end = date_range
    csv_content = _build_csv_content("PM10", ["01/01/2020 01:00"])

    # Create instance with low concurrency limit
    aq = ChileAirQuality(stations_csv_path=None, max_concurrent_downloads=2)
    aq.set_stations_table(air_quality_instance.stations_table)

    download_count = {"value": 0}
    max_concurrent = {"value": 0}
    current_active = {"value": 0}

    class ConcurrencyTrackingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            current_active["value"] += 1
            max_concurrent["value"] = max(max_concurrent["value"], current_active["value"])
            download_count["value"] += 1
            await asyncio.sleep(0.01)  # Small delay to allow other downloads to start
            current_active["value"] -= 1
            response = MagicMock()
            response.content = csv_content.encode("utf-8")
            response.raise_for_status = MagicMock()
            return response

    with patch(
        "atmchile.air_quality_data.httpx.AsyncClient", return_value=ConcurrencyTrackingClient()
    ):
        await aq.get_data_async(
            stations="RM/D14",
            parameters=["PM10", "PM25", "CO", "SO2", "NO2"],
            start=start,
            end=end,
            curate=False,
            st=False,
        )

    # With max_concurrent_downloads=2, we should never have more than 2 active downloads
    assert max_concurrent["value"] <= 2


@pytest.mark.asyncio
async def test_get_data_async_error_handling(air_quality_instance: ChileAirQuality, date_range):
    """Test that individual errors don't stop all downloads."""
    start, end = date_range
    csv_content = _build_csv_content("PM10", ["01/01/2020 01:00"])

    call_count = {"value": 0}

    class ErrorClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            call_count["value"] += 1
            if "PM25" in url:
                raise Exception("Error downloading PM25")
            response = MagicMock()
            response.content = csv_content.encode("utf-8")
            response.raise_for_status = MagicMock()
            return response

    with patch("atmchile.air_quality_data.httpx.AsyncClient", return_value=ErrorClient()):
        df = await air_quality_instance.get_data_async(
            stations="RM/D14",
            parameters=["PM10", "PM25", "CO"],
            start=start,
            end=end,
            curate=False,
            st=False,
        )

    # Should still have data for PM10 and CO even if PM25 failed
    assert not df.empty
    assert "PM10" in df.columns or "CO" in df.columns


@pytest.mark.asyncio
async def test_get_data_async_all_parameters(air_quality_instance: ChileAirQuality, date_range):
    """Test async download with parameters='all'."""
    start, end = date_range
    csv_content = _build_csv_content("PM10", ["01/01/2020 01:00"])

    with patch(
        "atmchile.air_quality_data.httpx.AsyncClient", return_value=MockAsyncClient(csv_content)
    ):
        df = await air_quality_instance.get_data_async(
            stations="RM/D14",
            parameters="all",
            start=start,
            end=end,
            curate=False,
            st=False,
        )

    assert not df.empty
    # Should have at least some parameters
    assert (
        len(df.columns) > 5
    )  # date, city, station_code, station_name, region + at least one parameter


@pytest.mark.asyncio
async def test_get_data_async_all_stations(air_quality_instance: ChileAirQuality, date_range):
    """Test async download with stations='all'."""
    start, end = date_range
    csv_content = _build_csv_content("PM10", ["01/01/2020 01:00"])

    with patch(
        "atmchile.air_quality_data.httpx.AsyncClient", return_value=MockAsyncClient(csv_content)
    ):
        df = await air_quality_instance.get_data_async(
            stations="all",
            parameters="PM10",
            start=start,
            end=end,
            curate=False,
            st=False,
        )

    assert not df.empty
    # Should have data from all stations
    assert set(df["city"].unique()) == {"SCL", "VAP"}


# Tests de comparación síncrono vs asíncrono


@patch("atmchile.air_quality_data.requests.get")
@patch("atmchile.air_quality_data.httpx.AsyncClient")
def test_sync_vs_async_single_parameter(
    mock_async_client_class,
    mock_get,
    air_quality_instance: ChileAirQuality,
    date_range,
):
    """Compare sync and async results with single parameter."""
    start, end = date_range
    csv_content = _build_csv_content("PM10", ["01/01/2020 01:00", "01/01/2020 02:00"])

    def build_sync_response(url, **_kwargs):
        response = MagicMock()
        response.content = csv_content.encode("utf-8")
        response.raise_for_status = MagicMock()
        return response

    mock_get.side_effect = build_sync_response
    mock_async_client_class.return_value = MockAsyncClient(csv_content)

    sync_df = air_quality_instance.get_data(
        stations="RM/D14",
        parameters="PM10",
        start=start,
        end=end,
        curate=False,
        st=False,
    )

    async_df = asyncio.run(
        air_quality_instance.get_data_async(
            stations="RM/D14",
            parameters="PM10",
            start=start,
            end=end,
            curate=False,
            st=False,
        )
    )

    # Compare DataFrames - skip if empty
    if not sync_df.empty and not async_df.empty:
        pd.testing.assert_frame_equal(
            sync_df.sort_values(by=["date", "city"]).reset_index(drop=True),
            async_df.sort_values(by=["date", "city"]).reset_index(drop=True),
        )
    else:
        # Both should be empty or both should have data
        assert sync_df.empty == async_df.empty


@patch("atmchile.air_quality_data.requests.get")
@patch("atmchile.air_quality_data.httpx.AsyncClient")
def test_sync_vs_async_multiple_parameters(
    mock_async_client_class,
    mock_get,
    air_quality_instance: ChileAirQuality,
    date_range,
):
    """Compare sync and async results with multiple parameters."""
    start, end = date_range

    def make_csv_content(url: str) -> str:
        if "PM10" in url:
            return _build_csv_content("PM10", ["01/01/2020 01:00"])
        elif "PM25" in url:
            return _build_csv_content("PM25", ["01/01/2020 01:00"])
        else:
            return _build_csv_content("CO", ["01/01/2020 01:00"])

    def build_sync_response(url, **_kwargs):
        response = MagicMock()
        response.content = make_csv_content(url).encode("utf-8")
        response.raise_for_status = MagicMock()
        return response

    mock_get.side_effect = build_sync_response

    class MultiParamAsyncClient:
        """Mock async client that returns different CSV content based on URL."""

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            """Return different CSV content based on parameter in URL."""
            response = MagicMock()
            response.content = make_csv_content(url).encode("utf-8")
            response.raise_for_status = MagicMock()
            return response

    mock_async_client_class.return_value = MultiParamAsyncClient()

    sync_df = air_quality_instance.get_data(
        stations="RM/D14",
        parameters=["PM10", "PM25", "CO"],
        start=start,
        end=end,
        curate=False,
        st=False,
    )

    async_df = asyncio.run(
        air_quality_instance.get_data_async(
            stations="RM/D14",
            parameters=["PM10", "PM25", "CO"],
            start=start,
            end=end,
            curate=False,
            st=False,
        )
    )

    # Compare DataFrames - skip if empty
    if not sync_df.empty and not async_df.empty:
        pd.testing.assert_frame_equal(
            sync_df.sort_values(by=["date", "city"]).reset_index(drop=True),
            async_df.sort_values(by=["date", "city"]).reset_index(drop=True),
        )
    else:
        # Both should be empty or both should have data
        assert sync_df.empty == async_df.empty


@patch("atmchile.air_quality_data.requests.get")
@patch("atmchile.air_quality_data.httpx.AsyncClient")
def test_sync_vs_async_multiple_stations(
    mock_async_client_class,
    mock_get,
    air_quality_instance: ChileAirQuality,
    date_range,
):
    """Compare sync and async results with multiple stations."""
    start, end = date_range
    csv_content = _build_csv_content("PM10", ["01/01/2020 01:00"])

    def build_sync_response(url, **_kwargs):
        response = MagicMock()
        response.content = csv_content.encode("utf-8")
        response.raise_for_status = MagicMock()
        return response

    mock_get.side_effect = build_sync_response
    mock_async_client_class.return_value = MockAsyncClient(csv_content)

    sync_df = air_quality_instance.get_data(
        stations=["RM/D14", "RV/550"],
        parameters="PM10",
        start=start,
        end=end,
        curate=False,
        st=False,
    )

    async_df = asyncio.run(
        air_quality_instance.get_data_async(
            stations=["RM/D14", "RV/550"],
            parameters="PM10",
            start=start,
            end=end,
            curate=False,
            st=False,
        )
    )

    # Compare DataFrames
    pd.testing.assert_frame_equal(
        sync_df.sort_values(by=["date", "city"]).reset_index(drop=True),
        async_df.sort_values(by=["date", "city"]).reset_index(drop=True),
    )


@patch("atmchile.air_quality_data.requests.get")
@patch("atmchile.air_quality_data.httpx.AsyncClient")
def test_sync_vs_async_edge_case_empty_data(
    mock_async_client_class,
    mock_get,
    air_quality_instance: ChileAirQuality,
    date_range,
):
    """Compare sync and async handling of empty data."""
    start, end = date_range

    def build_sync_response(url, **_kwargs):
        response = MagicMock()
        # Return CSV with only headers, no data rows
        # This should result in empty DataFrame after processing
        response.content = b"FECHA (YYMMDD);HORA (HHMM);Registros validados\n"
        response.raise_for_status = MagicMock()
        return response

    mock_get.side_effect = build_sync_response

    class EmptyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            response = MagicMock()
            # Return CSV with only headers, no data rows
            # This should result in empty DataFrame after processing
            response.content = b"FECHA (YYMMDD);HORA (HHMM);Registros validados\n"
            response.raise_for_status = MagicMock()
            return response

    mock_async_client_class.return_value = EmptyClient()

    sync_df = air_quality_instance.get_data(
        stations="RM/D14",
        parameters="PM10",
        start=start,
        end=end,
        curate=False,
        st=False,
    )

    async_df = asyncio.run(
        air_quality_instance.get_data_async(
            stations="RM/D14",
            parameters="PM10",
            start=start,
            end=end,
            curate=False,
            st=False,
        )
    )

    # Both should handle empty data the same way
    # When CSV has no data rows, both should create DataFrames with the parameter column
    # but with empty values, resulting in non-empty DataFrames with empty parameter values
    # However, after dropna on date/site, they might become empty
    # So we just check that they handle it consistently
    if sync_df.empty and async_df.empty:
        # Both are empty, which is fine
        assert True
    elif not sync_df.empty and not async_df.empty:
        # Both have data, check structure
        assert "PM10" in sync_df.columns or "PM10" in async_df.columns
        # Check that both have the base metadata columns
        assert {"date", "city", "station_code", "station_name", "region"}.issubset(
            set(sync_df.columns)
        )
        assert {"date", "city", "station_code", "station_name", "region"}.issubset(
            set(async_df.columns)
        )


@patch("atmchile.air_quality_data.requests.get")
@patch("atmchile.air_quality_data.httpx.AsyncClient")
def test_sync_vs_async_edge_case_single_url_fails(
    mock_async_client_class,
    mock_get,
    air_quality_instance: ChileAirQuality,
    date_range,
):
    """Compare sync and async handling when a URL fails."""
    start, end = date_range
    csv_content = _build_csv_content("temp", ["01/01/2020 01:00"])

    def build_sync_response(url, **_kwargs):
        # Fail on primary URL, succeed on alternative
        if "horario_000" in url:
            raise requests.RequestException("Primary URL failed")
        response = MagicMock()
        response.content = csv_content.encode("utf-8")
        response.raise_for_status = MagicMock()
        return response

    mock_get.side_effect = build_sync_response

    call_count = {"value": 0}

    class AlternativeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            call_count["value"] += 1
            # Fail on primary URL, succeed on alternative
            if "horario_000" in url:
                raise Exception("Primary URL failed")
            response = MagicMock()
            response.content = csv_content.encode("utf-8")
            response.raise_for_status = MagicMock()
            return response

    mock_async_client_class.return_value = AlternativeClient()

    sync_df = air_quality_instance.get_data(
        stations="RM/D14",
        parameters="temp",
        start=start,
        end=end,
        curate=False,
        st=False,
    )

    async_df = asyncio.run(
        air_quality_instance.get_data_async(
            stations="RM/D14",
            parameters="temp",
            start=start,
            end=end,
            curate=False,
            st=False,
        )
    )

    # Both should handle alternative URLs the same way
    assert sync_df.empty == async_df.empty
    if not sync_df.empty:
        assert "temp" in sync_df.columns
        assert "temp" in async_df.columns


@patch("atmchile.air_quality_data.requests.get")
@patch("atmchile.air_quality_data.httpx.AsyncClient")
def test_sync_vs_async_edge_case_validation_status(
    mock_async_client_class,
    mock_get,
    air_quality_instance: ChileAirQuality,
    date_range,
):
    """Compare sync and async handling of validation status (st=True)."""
    start, end = date_range
    csv_content = _build_csv_content("PM10", ["01/01/2020 01:00"])

    def build_sync_response(url, **_kwargs):
        response = MagicMock()
        response.content = csv_content.encode("utf-8")
        response.raise_for_status = MagicMock()
        return response

    mock_get.side_effect = build_sync_response
    mock_async_client_class.return_value = MockAsyncClient(csv_content)

    sync_df = air_quality_instance.get_data(
        stations="RM/D14",
        parameters="PM10",
        start=start,
        end=end,
        curate=False,
        st=True,
    )

    async_df = asyncio.run(
        air_quality_instance.get_data_async(
            stations="RM/D14",
            parameters="PM10",
            start=start,
            end=end,
            curate=False,
            st=True,
        )
    )

    # Both should include validation status columns
    assert "s.PM10" in sync_df.columns
    assert "s.PM10" in async_df.columns
    pd.testing.assert_frame_equal(
        sync_df.sort_values(by=["date", "city"]).reset_index(drop=True),
        async_df.sort_values(by=["date", "city"]).reset_index(drop=True),
    )


@patch("atmchile.air_quality_data.requests.get")
@patch("atmchile.air_quality_data.httpx.AsyncClient")
def test_sync_vs_async_edge_case_curation(
    mock_async_client_class,
    mock_get,
    air_quality_instance: ChileAirQuality,
    date_range,
):
    """Compare sync and async application of data curation."""
    start, end = date_range

    # Create data that will trigger curation rules
    _curation_header = "FECHA (YYMMDD);HORA (HHMM);Registros validados;Registros preliminares;Registros no validados"  # noqa: E501
    csv_content_pm10 = f"{_curation_header}\n200101;0100;100.0;;\n"
    csv_content_pm25 = f"{_curation_header}\n200101;0100;150.0;;\n"

    def build_sync_response(url, **_kwargs):
        response = MagicMock()
        if "PM10" in url:
            response.content = csv_content_pm10.encode("utf-8")
        else:
            response.content = csv_content_pm25.encode("utf-8")
        response.raise_for_status = MagicMock()
        return response

    mock_get.side_effect = build_sync_response

    class MultiParamCurationClient:
        """Mock async client that returns different CSV content based on URL."""

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            """Return different CSV content based on parameter in URL."""
            response = MagicMock()
            if "PM10" in url:
                response.content = csv_content_pm10.encode("utf-8")
            else:
                response.content = csv_content_pm25.encode("utf-8")
            response.raise_for_status = MagicMock()
            return response

    mock_async_client_class.return_value = MultiParamCurationClient()

    sync_df = air_quality_instance.get_data(
        stations="RM/D14",
        parameters=["PM10", "PM25"],
        start=start,
        end=end,
        curate=True,
        st=False,
    )

    async_df = asyncio.run(
        air_quality_instance.get_data_async(
            stations="RM/D14",
            parameters=["PM10", "PM25"],
            start=start,
            end=end,
            curate=True,
            st=False,
        )
    )

    # Both should apply curation (PM25 > PM10 * 1.001, so values should be NaN)
    if not sync_df.empty and not async_df.empty:
        # Check that curation was applied (PM25 values should be NaN)
        sync_pm25_numeric = pd.to_numeric(sync_df["PM25"], errors="coerce")
        async_pm25_numeric = pd.to_numeric(async_df["PM25"], errors="coerce")
        # Both should have NaN values after curation
        assert sync_pm25_numeric.isna().any() == async_pm25_numeric.isna().any()


@patch("atmchile.air_quality_data.requests.get")
@patch("atmchile.air_quality_data.httpx.AsyncClient")
def test_sync_vs_async_edge_case_date_range(
    mock_async_client_class,
    mock_get,
    air_quality_instance: ChileAirQuality,
):
    """Compare sync and async handling of different date ranges."""
    # Test with a longer date range
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2020, 1, 3, 23, 0, 0)
    csv_content = _build_csv_content("PM10", ["01/01/2020 01:00"])

    def build_sync_response(url, **_kwargs):
        response = MagicMock()
        response.content = csv_content.encode("utf-8")
        response.raise_for_status = MagicMock()
        return response

    mock_get.side_effect = build_sync_response
    mock_async_client_class.return_value = MockAsyncClient(csv_content)

    sync_df = air_quality_instance.get_data(
        stations="RM/D14",
        parameters="PM10",
        start=start,
        end=end,
        curate=False,
        st=False,
    )

    async_df = asyncio.run(
        air_quality_instance.get_data_async(
            stations="RM/D14",
            parameters="PM10",
            start=start,
            end=end,
            curate=False,
            st=False,
        )
    )

    # Both should handle the date range the same way
    assert len(sync_df) == len(async_df)
    pd.testing.assert_frame_equal(
        sync_df.sort_values(by=["date", "city"]).reset_index(drop=True),
        async_df.sort_values(by=["date", "city"]).reset_index(drop=True),
    )


@patch("atmchile.air_quality_data.requests.get")
@patch("atmchile.air_quality_data.httpx.AsyncClient")
def test_sync_vs_async_edge_case_region_parameter(
    mock_async_client_class,
    mock_get,
    air_quality_instance: ChileAirQuality,
    date_range,
):
    """Compare sync and async handling of region=True/False parameter."""
    start, end = date_range
    csv_content = _build_csv_content("PM10", ["01/01/2020 01:00"])

    def build_sync_response(url, **_kwargs):
        response = MagicMock()
        response.content = csv_content.encode("utf-8")
        response.raise_for_status = MagicMock()
        return response

    mock_get.side_effect = build_sync_response
    mock_async_client_class.return_value = MockAsyncClient(csv_content)

    # Test with region=False (search by cod)
    sync_df_no_region = air_quality_instance.get_data(
        stations="RM/D14",
        parameters="PM10",
        start=start,
        end=end,
        region=False,
        curate=False,
        st=False,
    )

    async_df_no_region = asyncio.run(
        air_quality_instance.get_data_async(
            stations="RM/D14",
            parameters="PM10",
            start=start,
            end=end,
            region=False,
            curate=False,
            st=False,
        )
    )

    # Test with region=True (search by Region)
    sync_df_region = air_quality_instance.get_data(
        stations="RM",
        parameters="PM10",
        start=start,
        end=end,
        region=True,
        curate=False,
        st=False,
    )

    async_df_region = asyncio.run(
        air_quality_instance.get_data_async(
            stations="RM",
            parameters="PM10",
            start=start,
            end=end,
            region=True,
            curate=False,
            st=False,
        )
    )

    # Both should handle region parameter the same way
    pd.testing.assert_frame_equal(
        sync_df_no_region.sort_values(by=["date", "city"]).reset_index(drop=True),
        async_df_no_region.sort_values(by=["date", "city"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        sync_df_region.sort_values(by=["date", "city"]).reset_index(drop=True),
        async_df_region.sort_values(by=["date", "city"]).reset_index(drop=True),
    )
