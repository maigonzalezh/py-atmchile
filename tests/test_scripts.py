"""Tests for atmchile.scripts module."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests as req

from atmchile.scripts import CSV_COLUMNS, _build_station_code, refresh_stations

# ---------------------------------------------------------------------------
# _build_station_code (pure function)
# ---------------------------------------------------------------------------


def test_build_station_code_regular_region() -> None:
    """Non-RM regions get an 'R' prefix."""
    assert _build_station_code("IV", "001") == "RIV/001"


def test_build_station_code_rm() -> None:
    """RM is used directly without an 'R' prefix."""
    assert _build_station_code("RM", "D14") == "RM/D14"


# ---------------------------------------------------------------------------
# refresh_stations — happy path
# ---------------------------------------------------------------------------


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_writes_csv(
    mock_get: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path: valid JSON response causes a CSV to be written."""
    monkeypatch.setattr("atmchile.scripts.DATA_DIR", tmp_path)
    mock_get.return_value.json.return_value = [
        {
            "key": "001",
            "region": "Región de Coquimbo",
            "latitud": "-30.0",
            "longitud": "-71.0",
            "nombre": "Test Station",
            "red": "SINCA",
            "regionindex": "1",
            "calificacion": "P",
            "empresa": "MMA",
            "comuna": "Coquimbo",
        }
    ]
    refresh_stations()

    csv_path = tmp_path / "sinca_stations.csv"
    assert csv_path.exists()

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == CSV_COLUMNS
    assert len(rows) == 1
    assert rows[0]["city"] == "Coquimbo"
    assert rows[0]["station_code"] == "RIV/001"
    assert rows[0]["region"] == "IV"


# ---------------------------------------------------------------------------
# refresh_stations — network error paths
# ---------------------------------------------------------------------------


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_connection_error_exits(
    mock_get: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """ConnectionError triggers sys.exit(1) with a descriptive message."""
    mock_get.side_effect = req.exceptions.ConnectionError("no route to host")
    with pytest.raises(SystemExit) as exc:
        refresh_stations()
    assert exc.value.code == 1
    assert "Could not connect" in capsys.readouterr().out


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_timeout_exits(mock_get: MagicMock) -> None:
    """Timeout triggers sys.exit(1)."""
    mock_get.side_effect = req.exceptions.Timeout()
    with pytest.raises(SystemExit) as exc:
        refresh_stations()
    assert exc.value.code == 1


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_http_error_exits(mock_get: MagicMock) -> None:
    """HTTPError triggers sys.exit(1)."""
    http_err = req.exceptions.HTTPError()
    http_err.response = MagicMock()
    http_err.response.status_code = 404
    mock_get.side_effect = http_err
    with pytest.raises(SystemExit) as exc:
        refresh_stations()
    assert exc.value.code == 1


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_generic_request_exception_exits(mock_get: MagicMock) -> None:
    """Generic RequestException triggers sys.exit(1)."""
    mock_get.side_effect = req.exceptions.RequestException("boom")
    with pytest.raises(SystemExit) as exc:
        refresh_stations()
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# refresh_stations — invalid response data
# ---------------------------------------------------------------------------


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_invalid_json_exits(mock_get: MagicMock) -> None:
    """Un-parseable JSON response triggers sys.exit(1)."""
    mock_get.return_value.json.side_effect = ValueError("not JSON")
    with pytest.raises(SystemExit) as exc:
        refresh_stations()
    assert exc.value.code == 1


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_non_list_response_exits(mock_get: MagicMock) -> None:
    """A JSON object (not a list) at the top level triggers sys.exit(1)."""
    mock_get.return_value.json.return_value = {"error": "unexpected"}
    with pytest.raises(SystemExit) as exc:
        refresh_stations()
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# refresh_stations — unknown region (warning, no exit)
# ---------------------------------------------------------------------------


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_unknown_region_warns(
    mock_get: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unrecognised region string prints a WARNING but does not exit."""
    monkeypatch.setattr("atmchile.scripts.DATA_DIR", tmp_path)
    mock_get.return_value.json.return_value = [
        {
            "key": "001",
            "region": "Región Desconocida XYZ",
            "latitud": "-30.0",
            "longitud": "-71.0",
            "nombre": "Test Station",
            "red": "SINCA",
            "regionindex": "1",
            "calificacion": "P",
            "empresa": "MMA",
            "comuna": "TestCity",
        }
    ]
    refresh_stations()
    out = capsys.readouterr().out
    assert "unrecognised region" in out
    assert (tmp_path / "sinca_stations.csv").exists()
