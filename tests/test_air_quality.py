"""High-value tests for ChileAirQuality."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from atmchile.air_quality_data import AirQualityDownloadStatus, ChileAirQuality


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
    aq = ChileAirQuality(stations_csv_path=None)
    aq.set_stations_table(mock_stations_df)
    return aq


@pytest.fixture
def date_range() -> tuple[datetime, datetime]:
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2020, 1, 1, 5, 0, 0)
    return start, end


def test_validate_request_expands_all(air_quality_instance: ChileAirQuality, date_range):
    start, end = date_range
    stations, params, _, _ = air_quality_instance._validate_and_prepare_request(
        stations="all",
        parameters="all",
        start=start,
        end=end,
    )

    assert stations == ["RM/D14", "RV/550"]  # station_code column values
    assert set(params) == set(air_quality_instance.PARAMETER_CODES.keys())


def test_validate_request_rejects_invalid_parameter(
    air_quality_instance: ChileAirQuality,
    date_range,
):
    start, end = date_range
    with pytest.raises(ValueError, match="is not available"):
        air_quality_instance._validate_and_prepare_request(
            stations="RM/D14",
            parameters="INVALID",
            start=start,
            end=end,
        )


def test_validate_request_rejects_invalid_dates(air_quality_instance: ChileAirQuality):
    with pytest.raises(ValueError, match="Start date must be before end date"):
        air_quality_instance._validate_and_prepare_request(
            stations="RM/D14",
            parameters="PM10",
            start=datetime(2020, 1, 2),
            end=datetime(2020, 1, 1),
        )


def test_init_reads_custom_csv(tmp_path):
    csv_file = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "station_name": ["Foo"],
            "city": ["BAR"],
            "station_code": ["CODE"],
            "latitude": [0.0],
            "longitude": [0.0],
            "region": ["I"],
        }
    ).to_csv(csv_file, index=False)

    aq = ChileAirQuality(stations_csv_path=str(csv_file))
    assert aq.stations_table is not None
    assert aq.stations_table.iloc[0]["station_code"] == "CODE"


def test_init_missing_csv_sets_none(tmp_path):
    missing = tmp_path / "missing.csv"
    aq = ChileAirQuality(stations_csv_path=str(missing))
    assert aq.stations_table is None


def test_get_stations_without_table_raises(tmp_path):
    missing = tmp_path / "missing.csv"
    aq = ChileAirQuality(stations_csv_path=str(missing))
    with pytest.raises(ValueError, match="Stations table has not been loaded"):
        aq.get_stations()


def test_validate_request_without_table_raises(tmp_path):
    missing = tmp_path / "missing.csv"
    aq = ChileAirQuality(stations_csv_path=str(missing))
    with pytest.raises(ValueError, match="Stations table has not been loaded"):
        aq._validate_and_prepare_request(
            stations="RM/D14",
            parameters="PM10",
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 2),
        )


def _build_param_df(start: datetime, end: datetime, parameter: str) -> pd.DataFrame:
    dates = pd.date_range(
        start=start.replace(hour=1, minute=0, second=0),
        end=end.replace(hour=23, minute=0, second=0),
        freq="h",
    )
    return pd.DataFrame({"date": dates.strftime(ChileAirQuality.SINCA_DATE_FORMAT), parameter: 1})


def test_get_data_merges_parameters(monkeypatch, air_quality_instance: ChileAirQuality):
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2020, 1, 1, 3, 0, 0)

    def fake_download(station_code, parameter, *_, **__):
        return _build_param_df(start, end, parameter)

    monkeypatch.setattr(air_quality_instance, "_download_parameter", fake_download)

    df = air_quality_instance.get_data(
        stations="RM/D14",
        parameters=["PM10", "PM25"],
        start=start,
        end=end,
        curate=False,
        st=False,
    )

    assert {"PM10", "PM25"}.issubset(df.columns)
    assert len(df) == len(
        pd.date_range(
            start=start.replace(hour=1, minute=0, second=0),
            end=end.replace(hour=23, minute=0, second=0),
            freq="h",
        )
    )


def test_get_data_region_lookup_uses_region(monkeypatch, air_quality_instance: ChileAirQuality):
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2020, 1, 1, 1, 0, 0)

    def fake_download(station_code, parameter, *_, **__):
        return _build_param_df(start, end, parameter)

    monkeypatch.setattr(air_quality_instance, "_download_parameter", fake_download)

    df = air_quality_instance.get_data(
        stations="RM",
        parameters="PM10",
        start=start,
        end=end,
        region=True,
        curate=False,
    )

    assert not df.empty
    assert df["city"].eq("SCL").all()  # SCL is the city for RM/D14


def test_get_data_handles_missing_station_column(
    air_quality_instance: ChileAirQuality,
    date_range,
):
    start, end = date_range
    broken_df = pd.DataFrame(
        {
            "city": ["SCL"],
            "station_code": ["BROKEN"],
            "latitude": [-1.0],
            "longitude": [-1.0],
            "station_name": ["Test Station"],
            "region": ["RM"],
        }
    )
    air_quality_instance.set_stations_table(broken_df)

    df = air_quality_instance.get_data(
        stations="BROKEN",
        parameters="PM10",
        start=start,
        end=end,
        curate=False,
    )
    # PM10 column is present but all values are NaN (download failed);
    # dl.PM10 carries DOWNLOAD_ERROR for all rows.
    assert not df.empty
    assert "PM10" in df.columns
    assert "dl.PM10" in df.columns
    assert df["PM10"].isna().all()
    assert (df["dl.PM10"] == AirQualityDownloadStatus.DOWNLOAD_ERROR).all()


def test_get_data_handles_station_processing_errors(
    monkeypatch, air_quality_instance: ChileAirQuality
):
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)

    def explode(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        air_quality_instance,
        "_combine_parameters_for_station",
        explode,
    )

    df = air_quality_instance.get_data(
        stations="RM/D14",
        parameters="PM10",
        start=start,
        end=end,
        curate=False,
    )
    assert df.empty


def test_get_data_invokes_curate(monkeypatch, air_quality_instance: ChileAirQuality):
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 1, 2)

    def fake_download(*_args, **_kwargs):
        return _build_param_df(start, end, "PM10")

    monkeypatch.setattr(air_quality_instance, "_download_parameter", fake_download)

    called = {"value": False}

    def fake_curate(df):
        called["value"] = True
        return df

    monkeypatch.setattr(air_quality_instance, "_curate_data", fake_curate)

    air_quality_instance.get_data(
        stations="RM/D14",
        parameters="PM10",
        start=start,
        end=end,
        curate=True,
    )
    assert called["value"] is True


def test_combine_parameters_merges_download_error_df(
    monkeypatch,
    air_quality_instance: ChileAirQuality,
    date_range,
):
    """_combine_parameters_for_station always merges, even on download error."""
    start, end = date_range

    error_df = air_quality_instance._create_empty_parameter_dataframe("PM10", start, end)

    def fake_download(*_args, **_kwargs):
        return error_df

    monkeypatch.setattr(air_quality_instance, "_download_parameter", fake_download)

    base_df = air_quality_instance._create_station_dataframe(
        city="SCL",
        station_code="RM/D14",
        station_name="Santiago",
        region="RM",
        dates_str=pd.Index(["01/01/2020 01:00"]),
    )
    result = air_quality_instance._combine_parameters_for_station(
        station_data=base_df,
        parameters_list=["PM10"],
        station_code="CODE",
        station_name="Test",
        start_datetime=start,
        end_datetime=end,
        st=False,
    )
    assert "PM10" in result.columns
    assert "dl.PM10" in result.columns
    assert (result["dl.PM10"] == AirQualityDownloadStatus.DOWNLOAD_ERROR).all()


def test_build_download_urls_supports_alternatives(
    air_quality_instance: ChileAirQuality, date_range
):
    start, end = date_range
    urls_temp = air_quality_instance._build_download_urls("STGO01", "temp", start, end)
    urls_pm10 = air_quality_instance._build_download_urls("STGO01", "PM10", start, end)

    assert len(urls_temp) == 2
    assert len(urls_pm10) == 1
    assert "temp" not in urls_pm10[0].lower()


def test_convert_numeric_columns_skips_metadata_and_lineage(air_quality_instance: ChileAirQuality):
    df = pd.DataFrame(
        {
            "date": ["01/01/2020 01:00"],
            "city": ["SCL"],
            "station_code": ["RM/D14"],
            "station_name": ["Santiago"],
            "region": ["RM"],
            "PM10": ["12"],
            "s.PM10": ["V"],
            "dl.PM10": [AirQualityDownloadStatus.OK],
        }
    )

    converted = air_quality_instance._convert_numeric_columns(df.copy())
    assert converted["PM10"].dtype.kind in {"i", "f"}
    assert converted["s.PM10"].dtype == object
    assert converted["dl.PM10"].dtype == object


def test_convert_numeric_columns_empty_returns_df(air_quality_instance: ChileAirQuality):
    empty = pd.DataFrame()
    assert air_quality_instance._convert_numeric_columns(empty).empty


def test_process_validation_status_prioritises_columns(air_quality_instance: ChileAirQuality):
    df = pd.DataFrame(
        {
            "Registros validados": ["10", "", ""],
            "Registros preliminares": ["", "11", ""],
            "Registros no validados": ["", "", "12"],
        }
    )

    statuses = air_quality_instance._process_validation_status(df, "PM10")
    assert statuses.tolist() == ["V", "PV", "NV"]


def test_parse_sinca_dates_handles_invalid_data(air_quality_instance: ChileAirQuality):
    good_df = pd.DataFrame({"FECHA": [200101], "HORA": [100]})
    dates = air_quality_instance._parse_sinca_dates(good_df, "FECHA", "HORA")
    assert dates.iloc[0] == "01/01/2020 01:00"

    bad_df = pd.DataFrame({"FECHA": ["bad"], "HORA": ["xx"]})
    fallback = air_quality_instance._parse_sinca_dates(bad_df, "FECHA", "HORA")
    assert fallback.tolist() == [""]


@patch("atmchile.air_quality_data.requests.get")
def test_download_parameter_returns_processed_data(
    mock_get,
    air_quality_instance: ChileAirQuality,
    date_range,
):
    start, end = date_range
    header = "FECHA (YYMMDD);HORA (HHMM);Registros validados;Registros preliminares;Registros no validados"  # noqa: E501
    csv_content = f"{header}\n200101;0100;12;;\n"
    mock_response = MagicMock()
    mock_response.content = csv_content.encode("utf-8")
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    result = air_quality_instance._download_parameter("STGO01", "PM10", start, end, st=True)
    assert "s.PM10" in result.columns
    assert "dl.PM10" in result.columns
    assert result["PM10"].iloc[0] == "12.0"
    assert result["s.PM10"].iloc[0] == "V"
    assert result["dl.PM10"].iloc[0] == AirQualityDownloadStatus.OK


@patch("atmchile.air_quality_data.requests.get")
def test_download_parameter_returns_download_error_for_incomplete_csv(
    mock_get,
    air_quality_instance: ChileAirQuality,
    date_range,
):
    start, end = date_range
    incomplete = "FECHA (YYMMDD)\n200101\n"
    mock_response = MagicMock()
    mock_response.content = incomplete.encode("utf-8")
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    result = air_quality_instance._download_parameter("STGO01", "PM10", start, end, st=False)
    assert isinstance(result, pd.DataFrame)
    assert "dl.PM10" in result.columns
    assert (result["dl.PM10"] == AirQualityDownloadStatus.DOWNLOAD_ERROR).all()


@patch("atmchile.air_quality_data.requests.get")
def test_download_parameter_retries_after_exception(
    mock_get,
    air_quality_instance: ChileAirQuality,
    date_range,
):
    start, end = date_range

    call_count = {"n": 0}

    def side_effect(*_args, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise requests.RequestException("boom")
        valid = "FECHA (YYMMDD);HORA (HHMM);Registros validados\n200101;0100;5\n"
        resp = MagicMock()
        resp.content = valid.encode("utf-8")
        resp.raise_for_status = MagicMock()
        return resp

    mock_get.side_effect = side_effect
    result = air_quality_instance._download_parameter("STGO01", "temp", start, end, st=False)
    assert not result.empty


@patch("atmchile.air_quality_data.requests.get")
def test_download_parameter_returns_download_error_when_all_urls_fail(
    mock_get,
    air_quality_instance: ChileAirQuality,
    date_range,
):
    start, end = date_range
    mock_get.side_effect = requests.RequestException("boom")
    result = air_quality_instance._download_parameter("STGO01", "PM10", start, end, st=False)
    empty_df = air_quality_instance._create_empty_parameter_dataframe("PM10", start, end)

    pd.testing.assert_frame_equal(result, empty_df)
    assert "dl.PM10" in result.columns
    assert (result["dl.PM10"] == AirQualityDownloadStatus.DOWNLOAD_ERROR).all()


def test_process_parameter_data_handles_meteorological_columns(
    air_quality_instance: ChileAirQuality,
):
    df = pd.DataFrame(
        {
            "FECHA (YYMMDD)": [200101],
            "HORA (HHMM)": [200],
            "temp": [25.5],
        }
    )
    result = air_quality_instance._process_parameter_data(df, "temp", st=False)
    assert result.iloc[0]["temp"] == "25.5"
    assert result.iloc[0]["dl.temp"] == AirQualityDownloadStatus.OK


def test_process_parameter_data_meteorological_without_data_cols(
    air_quality_instance: ChileAirQuality,
):
    df = pd.DataFrame(
        {
            "FECHA": [200101],
            "HORA": [100],
        }
    )
    result = air_quality_instance._process_parameter_data(df, "temp", st=False)
    assert result["temp"].iloc[0] == ""
    assert result["dl.temp"].iloc[0] == AirQualityDownloadStatus.EMPTY


def test_process_parameter_data_contaminant_without_validation(
    air_quality_instance: ChileAirQuality,
):
    df = pd.DataFrame(
        {
            "FECHA": [200101],
            "HORA": [100],
        }
    )
    result = air_quality_instance._process_parameter_data(df, "PM10", st=False)
    assert result["PM10"].iloc[0] == ""
    assert result["dl.PM10"].iloc[0] == AirQualityDownloadStatus.EMPTY


def test_create_empty_parameter_dataframe_structure(air_quality_instance: ChileAirQuality):
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2020, 1, 1, 3, 0, 0)
    df = air_quality_instance._create_empty_parameter_dataframe("PM10", start, end)
    assert "PM10" in df.columns
    assert "dl.PM10" in df.columns
    assert df["PM10"].eq("").all()
    assert (df["dl.PM10"] == AirQualityDownloadStatus.DOWNLOAD_ERROR).all()
    assert df["date"].iloc[0].startswith("01/01/2020")


def test_curate_data_enforces_rules(air_quality_instance: ChileAirQuality):
    df = pd.DataFrame(
        {
            "date": ["01/01/2020 01:00"],
            "city": ["SCL"],
            "station_code": ["RM/D14"],
            "station_name": ["Santiago"],
            "region": ["RM"],
            "NO": [120],
            "NO2": [80],
            "NOX": [100],
            "PM10": [50],
            "PM25": [55],
            "wd": [400],
            "RH": [150],
            "dl.NO": [AirQualityDownloadStatus.OK],
            "dl.NO2": [AirQualityDownloadStatus.OK],
            "dl.NOX": [AirQualityDownloadStatus.OK],
            "dl.PM10": [AirQualityDownloadStatus.OK],
            "dl.PM25": [AirQualityDownloadStatus.OK],
            "dl.wd": [AirQualityDownloadStatus.OK],
            "dl.RH": [AirQualityDownloadStatus.OK],
        }
    )
    curated = air_quality_instance._curate_data(df)

    assert pd.isna(curated.loc[0, "NO"])
    assert pd.isna(curated.loc[0, "NO2"])
    assert pd.isna(curated.loc[0, "NOX"])
    assert pd.isna(curated.loc[0, "PM10"])
    assert pd.isna(curated.loc[0, "PM25"])
    assert pd.isna(curated.loc[0, "wd"])
    assert pd.isna(curated.loc[0, "RH"])
    assert curated.loc[0, "dl.NO"] == AirQualityDownloadStatus.CURATED
    assert curated.loc[0, "dl.NO2"] == AirQualityDownloadStatus.CURATED
    assert curated.loc[0, "dl.NOX"] == AirQualityDownloadStatus.CURATED
    assert curated.loc[0, "dl.PM10"] == AirQualityDownloadStatus.CURATED
    assert curated.loc[0, "dl.PM25"] == AirQualityDownloadStatus.CURATED
    assert curated.loc[0, "dl.wd"] == AirQualityDownloadStatus.CURATED
    assert curated.loc[0, "dl.RH"] == AirQualityDownloadStatus.CURATED


def _make_exception_df():
    return pd.DataFrame(
        {
            "date": ["01/01/2020 01:00"],
            "city": ["SCL"],
            "station_code": ["RM/D14"],
            "station_name": ["Santiago"],
            "region": ["RM"],
            "NO": [100],
            "NO2": [50],
            "NOX": [120],
            "PM10": [50],
            "PM25": [20],
            "wd": [10],
            "RH": [10],
        }
    )


def _patch_to_numeric(monkeypatch, column_names):
    original = pd.to_numeric

    def broken(series, *args, **kwargs):
        if getattr(series, "name", "") in column_names:
            raise ValueError("boom")
        return original(series, *args, **kwargs)

    monkeypatch.setattr("atmchile.air_quality_data.pd.to_numeric", broken)


def test_curate_data_handles_nox_exceptions(monkeypatch, air_quality_instance: ChileAirQuality):
    _patch_to_numeric(monkeypatch, {"NO", "NO2", "NOX"})
    air_quality_instance._curate_data(_make_exception_df())


def test_curate_data_handles_pm_exceptions(monkeypatch, air_quality_instance: ChileAirQuality):
    _patch_to_numeric(monkeypatch, {"PM10", "PM25"})
    air_quality_instance._curate_data(_make_exception_df())


def test_curate_data_handles_wd_exceptions(monkeypatch, air_quality_instance: ChileAirQuality):
    _patch_to_numeric(monkeypatch, {"wd"})
    air_quality_instance._curate_data(_make_exception_df())


def test_curate_data_handles_rh_exceptions(monkeypatch, air_quality_instance: ChileAirQuality):
    _patch_to_numeric(monkeypatch, {"RH"})
    air_quality_instance._curate_data(_make_exception_df())


# ---------------------------------------------------------------------------
# Date boundary tests (unit)
# ---------------------------------------------------------------------------


def test_validate_request_same_start_end(air_quality_instance: ChileAirQuality) -> None:
    """start == end: validation passes and the date_range produces at least 1 timestamp."""
    dt = datetime(2023, 6, 15)
    _, _, start, end = air_quality_instance._validate_and_prepare_request(
        stations="RM/D14", parameters="PM25", start=dt, end=dt
    )
    assert start == end
    dates = pd.date_range(start=start.replace(hour=1), end=end.replace(hour=23), freq="h")
    assert len(dates) > 0


def test_validate_request_year_boundary(air_quality_instance: ChileAirQuality) -> None:
    """Dec→Jan crossover: validation passes and preserves distinct years."""
    _, _, start, end = air_quality_instance._validate_and_prepare_request(
        stations="RM/D14",
        parameters="PM25",
        start=datetime(2022, 12, 28),
        end=datetime(2023, 1, 3),
    )
    assert start.year == 2022
    assert end.year == 2023


def test_validate_request_end_is_now(air_quality_instance: ChileAirQuality) -> None:
    """end = datetime.now(): validation passes without exception."""
    from datetime import datetime as dt_cls

    _, _, start, end = air_quality_instance._validate_and_prepare_request(
        stations="RM/D14",
        parameters="PM25",
        start=datetime(2024, 1, 1),
        end=dt_cls.now(),
    )
    assert end >= start
