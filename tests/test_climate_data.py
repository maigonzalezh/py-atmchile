"""High-signal tests for ChileClimateData."""

from __future__ import annotations

import asyncio
import io
import zipfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from atmchile import ChileClimateData
from atmchile.climate_data import ClimateDownloadStatus


def _build_zip_bytes(
    station_code: str,
    year: int,
    parameter: str,
    start_dt: datetime,
    end_dt: datetime,
) -> bytes:
    """Create an in-memory ZIP matching the format delivered by the API."""
    dates = pd.date_range(start_dt, end_dt, freq="h")
    if dates.empty:
        dates = pd.date_range(datetime(year, 1, 1), periods=1, freq="h")

    data: dict[str, list] = {"Instante": dates}

    if parameter == "Temperatura":
        data["Ts"] = list(range(len(dates)))
    elif parameter == "PuntoRocio":
        data["Td"] = [12.0] * len(dates)
    elif parameter == "Humedad":
        data["HR"] = [50.0 + idx for idx in range(len(dates))]
    elif parameter == "Viento":
        data["dd"] = [180.0] * len(dates)
        data["ff"] = [5.0] * len(dates)
        data["VRB"] = [0.0] * len(dates)
    elif parameter == "PresionQFE":
        data["QFE"] = [950.0] * len(dates)
    elif parameter == "PresionQFF":
        data["QFF"] = [1013.25] * len(dates)

    buffer = io.BytesIO()
    filename = f"{station_code}_{year}_{parameter}_.csv"
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        csv_bytes = pd.DataFrame(data).to_csv(index=False, sep=";", decimal=".").encode("utf-8")
        zip_file.writestr(filename, csv_bytes)
    return buffer.getvalue()


@pytest.fixture
def mock_stations_df() -> pd.DataFrame:
    """Minimal station catalog for tests."""
    return pd.DataFrame(
        {
            "Código Nacional": ["180005", "200006", "330019"],
            "Nombre": ["Estación A", "Estación B", "Estación C"],
            "Latitud": [-33.45528, -20.53472, -33.45528],
            "Longitud": [-70.54222, -70.15000, -70.54222],
            "Region": ["RM", "II", "RM"],
        }
    )


@pytest.fixture
def climate_data_instance(mock_stations_df: pd.DataFrame) -> ChileClimateData:
    """ChileClimateData instance with the mock catalog."""
    ccd = ChileClimateData(stations_csv_path=None)
    ccd.set_stations_table(mock_stations_df)
    return ccd


@pytest.fixture
def single_day_range() -> tuple[datetime, datetime]:
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2020, 1, 1, 5, 0, 0)
    return start, end


@pytest.fixture
def two_year_range() -> tuple[datetime, datetime]:
    start = datetime(2019, 12, 31, 21, 0, 0)
    end = datetime(2020, 1, 1, 3, 0, 0)
    return start, end


def test_get_stations_returns_dataframe(climate_data_instance: ChileClimateData):
    stations = climate_data_instance.get_stations()
    assert not stations.empty
    assert {"Código Nacional", "Nombre"}.issubset(stations.columns)


def test_default_init_loads_packaged_table():
    ccd = ChileClimateData()
    assert ccd.stations_table is not None
    assert {"Código Nacional", "Nombre"}.issubset(ccd.stations_table.columns)
    assert len(ccd.stations_table) > 0


def test_init_with_custom_csv(tmp_path, mock_stations_df: pd.DataFrame):
    csv_file = tmp_path / "custom.csv"
    mock_stations_df.to_csv(csv_file, index=False)

    ccd = ChileClimateData(stations_csv_path=str(csv_file))
    actual = ccd.stations_table.copy()
    actual["Código Nacional"] = actual["Código Nacional"].astype(str)
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        mock_stations_df.reset_index(drop=True),
        check_dtype=False,
    )


def test_missing_table_blocks_access(tmp_path):
    missing_file = tmp_path / "absent.csv"
    ccd = ChileClimateData(stations_csv_path=str(missing_file))
    assert ccd.stations_table is None

    with pytest.raises(ValueError, match="Stations table has not been loaded"):
        ccd.get_stations()

    with pytest.raises(ValueError, match="Stations table has not been loaded"):
        ccd.get_data(
            stations="180005",
            parameters="Temperatura",
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 2),
        )


def test_validate_request_rejects_invalid_parameter(
    climate_data_instance: ChileClimateData, single_day_range
):
    start, end = single_day_range
    with pytest.raises(ValueError, match="is not available"):
        climate_data_instance._validate_and_prepare_request(
            stations="180005",
            parameters="InvalidParam",
            start=start,
            end=end,
        )


def test_validate_request_rejects_inverted_dates(
    climate_data_instance: ChileClimateData, single_day_range
):
    start, _ = single_day_range
    end = start - timedelta(hours=1)
    with pytest.raises(ValueError, match="Start date must be before end date"):
        climate_data_instance._validate_and_prepare_request(
            stations="180005",
            parameters="Temperatura",
            start=start,
            end=end,
        )


def test_get_data_region_without_matches_returns_empty(
    climate_data_instance: ChileClimateData, single_day_range
):
    start, end = single_day_range
    df = climate_data_instance.get_data(
        stations="XV",
        parameters="Temperatura",
        start=start,
        end=end,
        region=True,
    )
    assert df.empty


def _parse_request_url(url: str) -> tuple[str, int, str]:
    filename = url.rstrip("/").split("/")[-1]
    station, year, parameter, _ = filename.split("_", 3)
    return station, int(year), parameter


def _sync_response(content: bytes) -> MagicMock:
    """Build a minimal requests.get mock response."""
    response = MagicMock()
    response.content = content
    response.raise_for_status = MagicMock()
    return response


def _make_dummy_async_client(make_zip_fn):
    """Return a mock httpx.AsyncClient that builds ZIP responses via make_zip_fn."""

    class _DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            response = MagicMock()
            response.content = make_zip_fn(url)
            response.raise_for_status = MagicMock()
            return response

    return _DummyAsyncClient()


@patch("atmchile.climate_data.requests.get")
def test_get_data_combines_parameters_and_filters_dates(
    mock_get,
    climate_data_instance: ChileClimateData,
    two_year_range,
):
    start, end = two_year_range
    station_code = "180005"
    parameters = ["Temperatura", "Humedad"]

    def build_response(url: str, **_kwargs):
        code, year, parameter = _parse_request_url(url)
        year_start = max(start, datetime(year, 1, 1))
        year_end = min(end, datetime(year, 12, 31, 23, 0, 0))
        response = MagicMock()
        response.content = _build_zip_bytes(code, year, parameter, year_start, year_end)
        response.raise_for_status = MagicMock()
        return response

    mock_get.side_effect = build_response

    df = climate_data_instance.get_data(
        stations=station_code,
        parameters=parameters,
        start=start,
        end=end,
    )

    assert not df.empty
    assert df["date"].min() >= pd.Timestamp(start)
    assert df["date"].max() <= pd.Timestamp(end)
    assert {"Ts", "HR", "Nombre", "CodigoNacional"}.issubset(df.columns)
    assert df["CodigoNacional"].eq(station_code).all()
    assert df["date"].is_monotonic_increasing


@patch("atmchile.climate_data.requests.get")
def test_get_data_fills_nan_when_download_fails(
    mock_get,
    climate_data_instance: ChileClimateData,
    single_day_range,
):
    start, end = single_day_range

    mock_get.side_effect = requests.RequestException("boom")

    df = climate_data_instance.get_data(
        stations="180005",
        parameters="Temperatura",
        start=start,
        end=end,
    )

    assert not df.empty
    assert df["Ts"].isna().all()
    assert df["date"].min() >= pd.Timestamp(start)
    assert df["date"].max() <= pd.Timestamp(end)


@patch("atmchile.climate_data.requests.get")
@patch("atmchile.climate_data.httpx.AsyncClient")
def test_get_data_async_matches_sync_output(
    mock_async_client_class,
    mock_get,
    climate_data_instance: ChileClimateData,
    two_year_range,
):
    start, end = two_year_range
    station_code = "180005"
    parameters = ["Temperatura", "Humedad"]

    def make_zip(url: str) -> bytes:
        code, year, parameter = _parse_request_url(url)
        year_start = max(start, datetime(year, 1, 1))
        year_end = min(end, datetime(year, 12, 31, 23, 0, 0))
        return _build_zip_bytes(code, year, parameter, year_start, year_end)

    mock_get.side_effect = lambda url, **_: _sync_response(make_zip(url))
    mock_async_client_class.return_value = _make_dummy_async_client(make_zip)

    sync_df = climate_data_instance.get_data(
        stations=station_code,
        parameters=parameters,
        start=start,
        end=end,
    )

    async_df = asyncio.run(
        climate_data_instance.get_data_async(
            stations=station_code,
            parameters=parameters,
            start=start,
            end=end,
        )
    )

    pd.testing.assert_frame_equal(
        sync_df.reset_index(drop=True),
        async_df.reset_index(drop=True),
    )


@patch("atmchile.climate_data.requests.get")
@patch("atmchile.climate_data.httpx.AsyncClient")
def test_sync_vs_async_region_filter(
    mock_async_client_class,
    mock_get,
    climate_data_instance: ChileClimateData,
    single_day_range,
):
    start, end = single_day_range

    def make_zip(url: str) -> bytes:
        station, year, parameter = _parse_request_url(url)
        return _build_zip_bytes(station, year, parameter, start, end)

    mock_get.side_effect = lambda url, **_: _sync_response(make_zip(url))
    mock_async_client_class.return_value = _make_dummy_async_client(make_zip)

    sync_df = climate_data_instance.get_data(
        stations=["RM"], parameters="Temperatura", start=start, end=end, region=True
    )
    async_df = asyncio.run(
        climate_data_instance.get_data_async(
            stations=["RM"], parameters="Temperatura", start=start, end=end, region=True
        )
    )

    assert set(sync_df["CodigoNacional"].unique()) == {"180005", "330019"}
    pd.testing.assert_frame_equal(
        sync_df.sort_values("CodigoNacional").reset_index(drop=True),
        async_df.sort_values("CodigoNacional").reset_index(drop=True),
    )


def test_process_year_results_converts_errors_to_empty_frames(
    climate_data_instance: ChileClimateData,
):
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2021, 1, 2, 0, 0, 0)
    ok_df = pd.DataFrame(
        {
            "Instante": pd.date_range(start, datetime(2020, 1, 2), freq="h"),
            "Ts": 1.0,
        }
    )

    combined = climate_data_instance._process_year_results(
        unique_years=[2020, 2021],
        year_results=[ok_df, None],
        parameter="Temperatura",
        start_datetime=start,
        end_datetime=end,
    )

    assert not combined.empty
    future_rows = combined[combined["Instante"].dt.year == 2021]
    assert not future_rows.empty
    assert future_rows["Ts"].isna().all()


def test_process_year_results_handles_exception_entries(
    climate_data_instance: ChileClimateData,
):
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2021, 1, 2, 0, 0, 0)
    ok_df = pd.DataFrame(
        {
            "Instante": pd.date_range(start, periods=3, freq="h"),
            "Ts": 2.0,
        }
    )

    combined = climate_data_instance._process_year_results(
        unique_years=[2020, 2021],
        year_results=[ok_df, Exception("boom")],
        parameter="Temperatura",
        start_datetime=start,
        end_datetime=end,
    )

    year_2021 = combined[combined["Instante"].dt.year == 2021]
    assert not year_2021.empty
    assert year_2021["Ts"].isna().all()


@pytest.mark.parametrize(
    ("parameter", "expected_value_columns", "expected_dl_columns"),
    [
        ("Temperatura", ["Ts"], ["dl.Ts"]),
        ("PuntoRocio", ["Td"], ["dl.Td"]),
        ("Humedad", ["HR"], ["dl.HR"]),
        ("Viento", ["dd", "ff", "VRB"], ["dl.dd", "dl.ff", "dl.VRB"]),
        ("PresionQFE", ["QFE"], ["dl.QFE"]),
        ("PresionQFF", ["QFF"], ["dl.QFF"]),
    ],
)
def test_create_empty_dataframe_shapes_match_parameter(
    climate_data_instance: ChileClimateData,
    parameter: str,
    expected_value_columns: list[str],
    expected_dl_columns: list[str],
):
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2020, 1, 1, 2, 0, 0)
    df = climate_data_instance._create_empty_dataframe(parameter, start, end)

    expected_cols = {"Instante"} | set(expected_value_columns) | set(expected_dl_columns)
    assert set(df.columns) == expected_cols

    assert df[expected_value_columns].isna().all().all()
    assert (df[expected_dl_columns] == ClimateDownloadStatus.DOWNLOAD_ERROR).all().all()


def test_create_station_dataframe_sets_datetime(climate_data_instance: ChileClimateData):
    dates = pd.Series(["01-01-2020 00:00:00", "01-01-2020 01:00:00"])
    df = climate_data_instance._create_station_dataframe(
        station_code="180005",
        station_name="Estación A",
        latitude=-33.4,
        longitude=-70.5,
        dates_str=dates,
    )

    assert list(df["CodigoNacional"].unique()) == ["180005"]
    assert df["Nombre"].eq("Estación A").all()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].iloc[0] == pd.Timestamp("2020-01-01 00:00:00")


def _make_param_df(start: datetime, end: datetime, column: str) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="h")
    return pd.DataFrame({"Instante": dates, column: list(range(len(dates)))})


def test_combine_parameters_merges_columns(climate_data_instance: ChileClimateData):
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2020, 1, 1, 2, 0, 0)

    def fake_download(station_code, parameter, *_args):
        column = {"Temperatura": "Ts", "Humedad": "HR"}[parameter]
        return _make_param_df(start, end, column)

    combined = climate_data_instance._combine_parameters(
        parameters_list=["Temperatura", "Humedad"],
        station_code="180005",
        start_datetime=start,
        end_datetime=end,
        download_func=fake_download,
    )

    assert list(combined.columns) == ["date", "Ts", "HR"]
    assert combined["Ts"].iloc[-1] == 2
    assert combined["HR"].iloc[-1] == 2


def test_combine_parameters_async_merges_columns(
    climate_data_instance: ChileClimateData,
    monkeypatch: pytest.MonkeyPatch,
):
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2020, 1, 1, 1, 0, 0)

    async def fake_download_async(client, semaphore, station_code, parameter, *_args):
        column = {"Temperatura": "Ts", "Humedad": "HR"}[parameter]
        return _make_param_df(start, end, column)

    monkeypatch.setattr(
        climate_data_instance,
        "_download_parameter_async",
        fake_download_async,
    )

    result = asyncio.run(
        climate_data_instance._combine_parameters_async(
            parameters_list=["Temperatura", "Humedad"],
            station_code="180005",
            start_datetime=start,
            end_datetime=end,
        )
    )
    assert list(result.columns) == ["date", "Ts", "HR"]
    assert len(result) == 2


def test_finalize_dataframe_sorts_by_date(climate_data_instance: ChileClimateData):
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-01")],
            "CodigoNacional": ["2", "1"],
            "value": [1, 2],
        }
    )
    finalized = climate_data_instance._finalize_dataframe(df)
    assert list(finalized["date"]) == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
    ]
    assert list(finalized["value"]) == [2, 1]


def test_calculate_year_bounds_clips_range(climate_data_instance: ChileClimateData):
    start = datetime(2020, 6, 1, 0, 0, 0)
    end = datetime(2022, 3, 1, 0, 0, 0)

    year_start, year_end = climate_data_instance._calculate_year_bounds(2021, start, end)
    assert year_start == datetime(2021, 1, 1)
    assert year_end == datetime(2021, 12, 31, 23, 59, 59)

    partial_start, partial_end = climate_data_instance._calculate_year_bounds(2020, start, end)
    assert partial_start == start
    assert partial_end == datetime(2020, 12, 31, 23, 59, 59)


def test_build_download_urls_structure(climate_data_instance: ChileClimateData):
    url, csvname = climate_data_instance._build_download_urls("180005", 2020, "Humedad")
    assert url.endswith("180005_2020_Humedad_")
    assert csvname == "180005_2020_Humedad_.csv"


def test_process_year_data_returns_none_when_csv_missing(
    climate_data_instance: ChileClimateData,
):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("unexpected.csv", "foo,bar\n1,2")

    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    result = climate_data_instance._process_year_data(
        buffer.getvalue(),
        "expected.csv",
        start,
        end,
    )
    assert result is None


@patch("atmchile.climate_data.requests.get")
def test_download_parameter_injects_empty_dataframe_when_processing_fails(
    mock_get,
    climate_data_instance: ChileClimateData,
):
    start = datetime(2019, 12, 31, 0, 0, 0)
    end = datetime(2020, 1, 2, 0, 0, 0)

    response = MagicMock()
    response.content = b"unused"
    response.raise_for_status = MagicMock()
    mock_get.return_value = response

    valid_df = pd.DataFrame(
        {
            "Instante": pd.date_range(datetime(2020, 1, 1), periods=3, freq="h"),
            "Ts": 1.0,
        }
    )

    with patch.object(
        ChileClimateData,
        "_process_year_data",
        side_effect=[None, valid_df],
    ):
        df = climate_data_instance._download_parameter(
            station_code="180005",
            parameter="Temperatura",
            start_datetime=start,
            end_datetime=end,
        )

    year_2019 = df[df["Instante"].dt.year == 2019]
    year_2020 = df[df["Instante"].dt.year == 2020]
    assert year_2019["Ts"].isna().all()
    assert not year_2020["Ts"].isna().any()


def test_download_year_async_returns_none_on_http_error(
    climate_data_instance: ChileClimateData,
):
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2020, 1, 2, 0, 0, 0)

    class FailingClient:
        async def get(self, url: str):
            raise RuntimeError("network boom")

    async def run():
        semaphore = asyncio.Semaphore(1)
        return await climate_data_instance._download_year_async(
            client=FailingClient(),
            semaphore=semaphore,
            station_code="180005",
            parameter="Temperatura",
            year=2020,
            start_datetime=start,
            end_datetime=end,
        )

    result = asyncio.run(run())
    assert result is None


@patch("atmchile.climate_data.requests.get")
@patch("atmchile.climate_data.httpx.AsyncClient")
def test_sync_vs_async_year_download_fails(
    mock_async_client_class,
    mock_get,
    climate_data_instance: ChileClimateData,
    two_year_range,
):
    start, end = two_year_range  # 2019-12-31 21:00 → 2020-01-01 03:00

    def make_zip(url: str) -> bytes:
        station, year, parameter = _parse_request_url(url)
        year_start = max(start, datetime(year, 1, 1))
        year_end = min(end, datetime(year, 12, 31, 23, 0, 0))
        return _build_zip_bytes(station, year, parameter, year_start, year_end)

    def sync_side_effect(url: str, **_kwargs):
        _, year, _ = _parse_request_url(url)
        if year == 2019:
            raise requests.RequestException("boom")
        return _sync_response(make_zip(url))

    class FailingForYear2019:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def get(self, url: str):
            _, year, _ = _parse_request_url(url)
            if year == 2019:
                raise RuntimeError("boom")
            response = MagicMock()
            response.content = make_zip(url)
            response.raise_for_status = MagicMock()
            return response

    mock_get.side_effect = sync_side_effect
    mock_async_client_class.return_value = FailingForYear2019()

    sync_df = climate_data_instance.get_data(
        stations="180005", parameters="Temperatura", start=start, end=end
    )
    async_df = asyncio.run(
        climate_data_instance.get_data_async(
            stations="180005", parameters="Temperatura", start=start, end=end
        )
    )

    sync_2019 = sync_df[sync_df["date"].dt.year == 2019]
    async_2019 = async_df[async_df["date"].dt.year == 2019]
    assert sync_2019["Ts"].isna().all()
    assert async_2019["Ts"].isna().all()

    pd.testing.assert_frame_equal(
        sync_df.reset_index(drop=True),
        async_df.reset_index(drop=True),
    )


@patch("atmchile.climate_data.requests.get")
@patch("atmchile.climate_data.httpx.AsyncClient")
def test_sync_vs_async_multiple_stations(
    mock_async_client_class,
    mock_get,
    climate_data_instance: ChileClimateData,
    single_day_range,
):
    start, end = single_day_range

    def make_zip(url: str) -> bytes:
        station, year, parameter = _parse_request_url(url)
        return _build_zip_bytes(station, year, parameter, start, end)

    mock_get.side_effect = lambda url, **_: _sync_response(make_zip(url))
    mock_async_client_class.return_value = _make_dummy_async_client(make_zip)

    stations = ["180005", "200006"]

    sync_df = climate_data_instance.get_data(
        stations=stations, parameters="Temperatura", start=start, end=end
    )
    async_df = asyncio.run(
        climate_data_instance.get_data_async(
            stations=stations, parameters="Temperatura", start=start, end=end
        )
    )

    assert set(sync_df["CodigoNacional"].unique()) == set(stations)
    pd.testing.assert_frame_equal(
        sync_df.sort_values(["CodigoNacional", "date"]).reset_index(drop=True),
        async_df.sort_values(["CodigoNacional", "date"]).reset_index(drop=True),
    )
