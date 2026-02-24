"""Unit tests for atmchile.utils helpers."""

from atmchile.utils import convert_str_to_list, load_package_csv


def test_convert_str_to_list_handles_str_and_list():
    assert convert_str_to_list("foo") == ["foo"]
    assert convert_str_to_list(["bar", "baz"]) == ["bar", "baz"]


def test_load_package_csv_reads_packaged_data():
    df = load_package_csv("dmc_stations.csv")
    assert not df.empty
    assert {"Código Nacional", "Nombre"}.issubset(df.columns)
