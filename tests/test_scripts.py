"""Tests for atmchile.scripts module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests as req

from atmchile.scripts import _build_station_code, refresh_stations


# ---------------------------------------------------------------------------
# _build_station_code (pure function)
# ---------------------------------------------------------------------------


def test_build_station_code_regular_region():
    """Non-RM regions get an 'R' prefix."""
    assert _build_station_code("IV", "001") == "RIV/001"


def test_build_station_code_rm():
    """RM is used directly without an 'R' prefix."""
    assert _build_station_code("RM", "D14") == "RM/D14"


# ---------------------------------------------------------------------------
# refresh_stations — happy path
# ---------------------------------------------------------------------------


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_writes_csv(mock_get, tmp_path, monkeypatch):
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
    assert (tmp_path / "sinca_stations.csv").exists()


# ---------------------------------------------------------------------------
# refresh_stations — network error paths
# ---------------------------------------------------------------------------


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_connection_error_exits(mock_get, capsys):
    """ConnectionError triggers sys.exit(1) with a descriptive message."""
    mock_get.side_effect = req.exceptions.ConnectionError("no route to host")
    with pytest.raises(SystemExit) as exc:
        refresh_stations()
    assert exc.value.code == 1
    assert "Could not connect" in capsys.readouterr().out


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_timeout_exits(mock_get):
    """Timeout triggers sys.exit(1)."""
    mock_get.side_effect = req.exceptions.Timeout()
    with pytest.raises(SystemExit) as exc:
        refresh_stations()
    assert exc.value.code == 1


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_http_error_exits(mock_get):
    """HTTPError triggers sys.exit(1)."""
    http_err = req.exceptions.HTTPError()
    http_err.response = MagicMock()
    http_err.response.status_code = 404
    mock_get.side_effect = http_err
    with pytest.raises(SystemExit) as exc:
        refresh_stations()
    assert exc.value.code == 1


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_generic_request_exception_exits(mock_get):
    """Generic RequestException triggers sys.exit(1)."""
    mock_get.side_effect = req.exceptions.RequestException("boom")
    with pytest.raises(SystemExit) as exc:
        refresh_stations()
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# refresh_stations — invalid response data
# ---------------------------------------------------------------------------


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_invalid_json_exits(mock_get):
    """Un-parseable JSON response triggers sys.exit(1)."""
    mock_get.return_value.json.side_effect = ValueError("not JSON")
    with pytest.raises(SystemExit) as exc:
        refresh_stations()
    assert exc.value.code == 1


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_non_list_response_exits(mock_get):
    """A JSON object (not a list) at the top level triggers sys.exit(1)."""
    mock_get.return_value.json.return_value = {"error": "unexpected"}
    with pytest.raises(SystemExit) as exc:
        refresh_stations()
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# refresh_stations — unknown region (warning, no exit)
# ---------------------------------------------------------------------------


@patch("atmchile.scripts.requests.get")
def test_refresh_stations_unknown_region_warns(mock_get, tmp_path, monkeypatch, capsys):
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
