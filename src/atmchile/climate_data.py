from __future__ import annotations

import asyncio
import io
import os
import sys
import zipfile
from collections.abc import Callable
from datetime import datetime

import httpx
import numpy as np
import pandas as pd
import requests

from atmchile.utils import convert_str_to_list, load_package_csv

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """Backport of StrEnum for Python < 3.11."""


class ClimateDownloadStatus(StrEnum):
    """Status of a climate parameter measurement for a given datetime row."""

    OK = "ok"
    EMPTY = "empty"
    DOWNLOAD_ERROR = "download_error"


class ChileClimateData:
    """
    Class to obtain climate data from monitoring stations in Chile.

    This class allows downloading and processing climate data from different
    meteorological stations in Chile, with support for multiple parameters
    and custom date ranges.

    Available parameters (synoptic observations from DMC):
        - ``Temperatura``  — Air temperature                  [°C]  → column ``Ts``
        - ``PuntoRocio``   — Dew point temperature            [°C]  → column ``Td``
        - ``Humedad``      — Relative humidity                [%]   → column ``HR``
        - ``Viento``       — Wind direction [°] / speed [m/s] / variability flag
                             → columns ``dd``, ``ff``, ``VRB``
        - ``PresionQFE``   — Station-level pressure (QFE)    [hPa] → column ``QFE``
        - ``PresionQFF``   — Sea-level pressure (QFF)        [hPa] → column ``QFF``

    Examples:
        >>> from datetime import datetime
        >>> ccd = ChileClimateData()
        >>> data = ccd.get_data(
        ...     stations="180005",
        ...     parameters=["Temperatura", "Humedad"],
        ...     start=datetime(2020, 1, 1, 0, 0, 0),
        ...     end=datetime(2020, 12, 31, 23, 0, 0),
        ... )
    """

    DATE_FORMAT: str = "%d-%m-%Y %H:%M:%S"

    url_base: str
    available_parameters: list[str]
    max_concurrent_downloads: int
    stations_table: pd.DataFrame | None

    def __init__(
        self,
        stations_csv_path: str | None = None,
        max_concurrent_downloads: int = 5,
    ) -> None:
        """
        Initialize the class with the stations table.

        Args:
            stations_csv_path: Path to the CSV file with station information.
                               If None, the packaged table is used or can be
                               set manually using the set_stations_table() method.
            max_concurrent_downloads: Maximum number of simultaneous downloads for
                                     asynchronous operations (default: 5).
        """
        self.url_base = "https://climatologia.meteochile.gob.cl/application/datos/getDatosSaclim/"
        self.available_parameters = [
            "Temperatura",
            "PuntoRocio",
            "Humedad",
            "Viento",
            "PresionQFE",
            "PresionQFF",
        ]
        self.max_concurrent_downloads = max_concurrent_downloads

        if stations_csv_path:
            if os.path.exists(stations_csv_path):
                self.stations_table = pd.read_csv(stations_csv_path, encoding="utf-8")
            else:
                self.stations_table = None
        else:
            self.stations_table = load_package_csv("dmc_stations.csv")

    def set_stations_table(self, dataframe: pd.DataFrame) -> None:
        """
        Manually set the stations table.

        Args:
            dataframe: DataFrame with station information. Must contain
                      columns: "Código Nacional", "Nombre", "Latitud", "Longitud",
                      and optionally "Region".
        """
        self.stations_table = dataframe

    def get_stations(self) -> pd.DataFrame:
        """
        Return the table with information of all available stations.

        Returns:
            DataFrame with station information

        Raises:
            ValueError: If the stations table has not been loaded
        """
        if self.stations_table is None:
            raise ValueError("Stations table has not been loaded")
        return self.stations_table

    def _validate_and_prepare_request(
        self,
        stations: str | list[str],
        parameters: str | list[str],
        start: datetime,
        end: datetime,
    ) -> tuple[list[str], list[str], datetime, datetime]:
        """
        Validate and prepare the parameters for a data request.

        Args:
            stations: Station code(s) or administrative region(s)
            parameters: Climate parameter(s) to query
            start: Start date and time (datetime object)
            end: End date and time (datetime object)

        Returns:
            Tuple with (stations_list, parameters_list, start_datetime, end_datetime)

        Raises:
            ValueError: If the stations table has not been loaded, if dates
                       are invalid, or if any parameter is not available.
        """
        if self.stations_table is None:
            raise ValueError("Stations table has not been loaded")

        stations_list = convert_str_to_list(stations)
        parameters_list = convert_str_to_list(parameters)

        if end < start:
            raise ValueError("Start date must be before end date")

        for param in parameters_list:
            if param not in self.available_parameters:
                raise ValueError(
                    f"Parameter '{param}' is not available. "
                    f"Available parameters: {self.available_parameters}"
                )

        return stations_list, parameters_list, start, end

    def get_data(
        self,
        stations: str | list[str],
        parameters: str | list[str],
        start: datetime,
        end: datetime,
        region: bool = False,
    ) -> pd.DataFrame:
        """
        Get climate data from the specified stations.

        Important note about downloads:
            Downloads from the server are performed by full year. If a partial
            range is requested (e.g., January to March 2023), the full year 2023
            will be downloaded and then filtered to return only the specified
            range between start and end.

        Args:
            stations: Station code(s) or administrative region(s)
            parameters: Climate parameter(s) to query
            start: Start date and time (datetime object)
                   The full year containing this date will be downloaded
            end: End date and time (datetime object)
                 The full year containing this date will be downloaded
            region: If True, allows entering the administrative region
                   instead of the station code

        Returns:
            DataFrame with the requested climate data, filtered to include
            only the specified range between start and end.

        Examples:
            >>> from datetime import datetime
            >>> ccd = ChileClimateData()
            >>> data = ccd.get_data(
            ...     stations="180005",
            ...     parameters=["Temperatura", "Humedad"],
            ...     start=datetime(2020, 1, 15, 0, 0, 0),
            ...     end=datetime(2020, 12, 31, 23, 0, 0),
            ... )

        Raises:
            ValueError: If the stations table has not been loaded, if dates
                       are invalid, or if any parameter is not available.
        """
        stations_list, parameters_list, start_datetime, end_datetime = (
            self._validate_and_prepare_request(stations, parameters, start, end)
        )

        dates = pd.date_range(start=start_datetime, end=end_datetime, freq="h")
        dates_str = dates.strftime(self.DATE_FORMAT)

        search_column = "Region" if region else "Código Nacional"

        assert self.stations_table is not None  # validated in _validate_and_prepare_request
        # Accumulate DataFrames in a list for efficient concatenation at the end
        dataframes_list = []

        for station in stations_list:
            matching_stations = self.stations_table[
                self.stations_table[search_column].astype(str) == str(station)
            ]

            for _, station_row in matching_stations.iterrows():
                station_code = station_row["Código Nacional"]
                station_name = station_row["Nombre"]
                latitude = station_row["Latitud"]
                longitude = station_row["Longitud"]

                station_data = self._create_station_dataframe(
                    station_code, station_name, latitude, longitude, dates_str
                )

                combined_param_df = self._combine_parameters(
                    parameters_list,
                    station_code,
                    start_datetime,
                    end_datetime,
                    self._download_parameter,
                )

                combined_params_with_station_data = pd.merge(
                    combined_param_df,
                    station_data,
                    left_on="date",
                    right_on="date",
                    how="left",
                )
                dataframes_list.append(combined_params_with_station_data)

        if dataframes_list:
            total_data = pd.concat(dataframes_list, ignore_index=True)
        else:
            total_data = pd.DataFrame()

        return self._finalize_dataframe(total_data)

    async def get_data_async(
        self,
        stations: str | list[str],
        parameters: str | list[str],
        start: datetime,
        end: datetime,
        region: bool = False,
    ) -> pd.DataFrame:
        """
        Get climate data from the specified stations asynchronously.

        Asynchronous version that allows parallel downloads to improve performance
        when querying multiple years of data.

        Important note about downloads:
            Downloads from the server are performed by full year. If a partial
            range is requested (e.g., January to March 2023), the full year 2023
            will be downloaded and then filtered to return only the specified
            range between start and end. Downloads for multiple years are
            performed in parallel according to the limit configured in
            max_concurrent_downloads.

        Args:
            stations: Station code(s) or administrative region(s)
            parameters: Climate parameter(s) to query
            start: Start date and time (datetime object)
                   The full year containing this date will be downloaded
            end: End date and time (datetime object)
                 The full year containing this date will be downloaded
            region: If True, allows entering the administrative region
                   instead of the station code

        Returns:
            DataFrame with the requested climate data, filtered to include
            only the specified range between start and end.

        Examples:
            >>> import asyncio
            >>> from datetime import datetime
            >>> ccd = ChileClimateData()
            >>> data = asyncio.run(ccd.get_data_async(
            ...     stations="180005",
            ...     parameters=["Temperatura", "Humedad"],
            ...     start=datetime(2020, 1, 15, 0, 0, 0),
            ...     end=datetime(2020, 12, 31, 23, 0, 0),
            ... ))

        Raises:
            ValueError: If the stations table has not been loaded, if dates
                       are invalid, or if any parameter is not available.
        """
        stations_list, parameters_list, start_datetime, end_datetime = (
            self._validate_and_prepare_request(stations, parameters, start, end)
        )

        dates = pd.date_range(start=start_datetime, end=end_datetime, freq="h")
        dates_str = dates.strftime(self.DATE_FORMAT)

        search_column = "Region" if region else "Código Nacional"

        assert self.stations_table is not None  # validated in _validate_and_prepare_request
        dataframes_list = []

        for station in stations_list:
            matching_stations = self.stations_table[
                self.stations_table[search_column].astype(str) == str(station)
            ]

            for _, station_row in matching_stations.iterrows():
                station_code = station_row["Código Nacional"]
                station_name = station_row["Nombre"]
                latitude = station_row["Latitud"]
                longitude = station_row["Longitud"]

                station_data = self._create_station_dataframe(
                    station_code, station_name, latitude, longitude, dates_str
                )

                combined_param_df = await self._combine_parameters_async(
                    parameters_list,
                    station_code,
                    start_datetime,
                    end_datetime,
                )

                combined_params_with_station_data = pd.merge(
                    combined_param_df,
                    station_data,
                    left_on="date",
                    right_on="date",
                    how="left",
                )
                dataframes_list.append(combined_params_with_station_data)

        if dataframes_list:
            total_data = pd.concat(dataframes_list, ignore_index=True)
        else:
            total_data = pd.DataFrame()

        return self._finalize_dataframe(total_data)

    def _create_station_dataframe(
        self,
        station_code: str,
        station_name: str,
        latitude: float,
        longitude: float,
        dates_str: pd.Index,
    ) -> pd.DataFrame:
        """
        Create a DataFrame with station information and dates.

        Args:
            station_code: Station code
            station_name: Station name
            latitude: Station latitude
            longitude: Station longitude
            dates_str: Index with dates formatted as strings

        Returns:
            DataFrame with station information
        """
        station_data = pd.DataFrame(
            {
                "date": dates_str,
                "Nombre": station_name,
                "CodigoNacional": station_code,
                "Latitud": latitude,
                "Longitud": longitude,
            }
        )
        station_data["date"] = pd.to_datetime(station_data["date"], dayfirst=True)
        return station_data

    def _combine_parameters(
        self,
        parameters_list: list[str],
        station_code: str,
        start_datetime: datetime,
        end_datetime: datetime,
        download_func: Callable[[str, str, datetime, datetime], pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Combine multiple parameters into a single DataFrame.

        Args:
            parameters_list: List of parameters to combine
            station_code: Station code
            start_datetime: Start date
            end_datetime: End date
            download_func: Function to download parameters (synchronous or asynchronous)

        Returns:
            DataFrame with all parameters combined
        """
        combined_param_df = pd.DataFrame()
        combined_param_df["date"] = pd.date_range(start=start_datetime, end=end_datetime, freq="h")

        for parameter in parameters_list:
            if parameter in self.available_parameters:
                param_df = download_func(station_code, parameter, start_datetime, end_datetime)

                combined_param_df = pd.merge(
                    combined_param_df,
                    param_df,
                    left_on="date",
                    right_on="Instante",
                    how="left",
                )

                combined_param_df = combined_param_df.drop(columns=["Instante"])

        return combined_param_df

    async def _combine_parameters_async(
        self,
        parameters_list: list[str],
        station_code: str,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> pd.DataFrame:
        """
        Combine multiple parameters into a single DataFrame (asynchronous version).

        Args:
            parameters_list: List of parameters to combine
            station_code: Station code
            start_datetime: Start date
            end_datetime: End date

        Returns:
            DataFrame with all parameters combined
        """
        combined_param_df = pd.DataFrame()
        combined_param_df["date"] = pd.date_range(start=start_datetime, end=end_datetime, freq="h")

        valid_params = [p for p in parameters_list if p in self.available_parameters]

        async with httpx.AsyncClient(timeout=30.0) as client:
            semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
            parameter_tasks = [
                self._download_parameter_async(
                    client, semaphore, station_code, p, start_datetime, end_datetime
                )
                for p in valid_params
            ]
            parameter_results = await asyncio.gather(*parameter_tasks, return_exceptions=True)

        for parameter, param_df in zip(valid_params, parameter_results):
            if isinstance(param_df, BaseException):
                print(f"Error downloading {parameter}: {param_df}")
                param_df = self._create_empty_dataframe(parameter, start_datetime, end_datetime)

            combined_param_df = pd.merge(
                combined_param_df,
                param_df,
                left_on="date",
                right_on="Instante",
                how="left",
            )
            combined_param_df = combined_param_df.drop(columns=["Instante"])

        return combined_param_df

    def _finalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort and clean the final DataFrame.

        Args:
            df: DataFrame to process

        Returns:
            Sorted DataFrame without duplicates
        """
        if df.empty:
            return df

        sort_columns = ["date", "CodigoNacional"] if "date" in df.columns else ["CodigoNacional"]
        return df.sort_values(sort_columns).reset_index(drop=True)

    def _calculate_year_bounds(
        self, year: int, start_datetime: datetime, end_datetime: datetime
    ) -> tuple[datetime, datetime]:
        """
        Calculate date bounds for a specific year within a range.

        Args:
            year: Year to process
            start_datetime: Start date of the full range
            end_datetime: End date of the full range

        Returns:
            Tuple with (year_start, year_end)
        """
        year_start = max(start_datetime, datetime(year, 1, 1))
        year_end = min(end_datetime, datetime(year, 12, 31, 23, 59, 59))
        return year_start, year_end

    def _build_download_urls(self, station_code: str, year: int, parameter: str) -> tuple[str, str]:
        """
        Build the URL and CSV filename for downloading data.

        Args:
            station_code: Station code
            year: Year to download
            parameter: Parameter to download

        Returns:
            Tuple with (url, csvname)
        """
        url = f"{self.url_base}{station_code}_{year}_{parameter}_"
        csvname = f"{station_code}_{year}_{parameter}_.csv"
        return url, csvname

    def _process_year_results(
        self,
        unique_years: list[int],
        year_results: list[pd.DataFrame | None | BaseException],
        parameter: str,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> pd.DataFrame:
        """
        Process year download results and combine them into a DataFrame.

        Args:
            unique_years: List of processed years
            year_results: List of results (DataFrames, Exceptions or None)
            parameter: Parameter name
            start_datetime: Start date
            end_datetime: End date

        Returns:
            Combined DataFrame with all years
        """
        year_dataframes = []

        for year, result in zip(unique_years, year_results):
            if isinstance(result, BaseException):
                print(f"Error downloading {parameter} for year {year}: {result}")
                year_start, year_end = self._calculate_year_bounds(
                    year, start_datetime, end_datetime
                )
                empty_df = self._create_empty_dataframe(parameter, year_start, year_end)
                year_dataframes.append(empty_df)
            elif result is not None:
                year_dataframes.append(result)
            else:
                print(f"Error downloading {parameter} for year {year}: Error processing data")
                year_start, year_end = self._calculate_year_bounds(
                    year, start_datetime, end_datetime
                )
                empty_df = self._create_empty_dataframe(parameter, year_start, year_end)
                year_dataframes.append(empty_df)

        if year_dataframes:
            combined_df = pd.concat(year_dataframes, ignore_index=True)
        else:
            combined_df = pd.DataFrame()

        if not combined_df.empty:
            combined_df = combined_df.sort_values("Instante")
            combined_df = combined_df.drop_duplicates(subset=["Instante"], keep="first")

        return combined_df

    def _process_year_data(
        self,
        zip_content: bytes,
        csvname: str,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> pd.DataFrame | None:
        """
        Process year data from a ZIP file.

        Extracts and processes the CSV from the ZIP, applying date filters.

        Args:
            zip_content: ZIP file content in bytes
            csvname: CSV filename inside the ZIP
            start_datetime: Start date and time of the range
            end_datetime: End date and time of the range

        Returns:
            Processed DataFrame or None if there is an error
        """
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
                with z.open(csvname) as csvfile:
                    df = pd.read_csv(csvfile, sep=";", decimal=".", encoding="utf-8")
                    if "CodigoNacional" in df.columns:
                        df = df.drop(columns=["CodigoNacional"])

                    df["Instante"] = pd.to_datetime(df["Instante"])
                    mask = (df["Instante"] >= start_datetime) & (df["Instante"] <= end_datetime)
                    df = df[mask]  # type: ignore[assignment]  # pandas-stubs: df[bool_mask] -> DataFrame
                    return df
        except Exception:
            return None

    def _download_parameter(
        self,
        station_code: str,
        parameter: str,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> pd.DataFrame:
        """
        Download data for a specific parameter for a station.

        Automatically downloads all necessary years based on the specified
        date range. If downloading a year fails, creates a DataFrame with
        NaN values for that period.

        Args:
            station_code: Meteorological station code
            parameter: Climate parameter name to download
            start_datetime: Start date and time of the requested range
            end_datetime: End date and time of the requested range

        Returns:
            DataFrame with parameter data for the entire date range.
            Contains the "Instante" column (datetime) and parameter-specific
            columns. If any year has no available data, it is filled with NaN.
        """
        unique_years = sorted(set(range(start_datetime.year, end_datetime.year + 1)))

        year_dataframes = []

        for year in unique_years:
            url, csvname = self._build_download_urls(station_code, year, parameter)
            print("File URL", url)

            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                df = self._process_year_data(
                    response.content, csvname, start_datetime, end_datetime
                )

                if df is not None:
                    df = self._add_status_columns(df)
                    year_dataframes.append(df)
                else:
                    raise ValueError("Error processing data")

            except Exception as e:
                print(f"Error downloading {parameter} for year {year}: {e}")
                year_start, year_end = self._calculate_year_bounds(
                    year, start_datetime, end_datetime
                )
                empty_df = self._create_empty_dataframe(parameter, year_start, year_end)
                year_dataframes.append(empty_df)

        if year_dataframes:
            combined_df = pd.concat(year_dataframes, ignore_index=True)
        else:
            combined_df = pd.DataFrame()

        if not combined_df.empty:
            combined_df = combined_df.sort_values("Instante")
            combined_df = combined_df.drop_duplicates(subset=["Instante"], keep="first")

        return combined_df

    async def _download_year_async(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        station_code: str,
        parameter: str,
        year: int,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> pd.DataFrame | None:
        """
        Download data for a specific year using httpx asynchronously.

        Args:
            client: Async HTTP client from httpx
            semaphore: Semaphore to limit concurrency
            station_code: Meteorological station code
            parameter: Climate parameter name
            year: Year to download
            start_datetime: Start date and time of the range
            end_datetime: End date and time of the range

        Returns:
            DataFrame with year data or None if there is an error
        """
        async with semaphore:
            url, csvname = self._build_download_urls(station_code, year, parameter)
            print("File URL", url)

            try:
                response = await client.get(url)
                response.raise_for_status()

                df = self._process_year_data(
                    response.content, csvname, start_datetime, end_datetime
                )
                if df is not None:
                    df = self._add_status_columns(df)
                return df

            except Exception as e:
                print(f"Error downloading {parameter} for year {year}: {e}")
                return None

    async def _download_parameter_async(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        station_code: str,
        parameter: str,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> pd.DataFrame:
        """
        Download data for a specific parameter asynchronously.

        Automatically downloads all necessary years in parallel based on the
        specified date range. If downloading a year fails, creates a DataFrame
        with NaN values for that period.

        Args:
            client: Shared async HTTP client
            semaphore: Shared semaphore to limit global concurrency
            station_code: Meteorological station code
            parameter: Climate parameter name to download
            start_datetime: Start date and time of the requested range
            end_datetime: End date and time of the requested range

        Returns:
            DataFrame with parameter data for the entire date range.
            Contains the "Instante" column (datetime) and parameter-specific
            columns. If any year has no available data, it is filled with NaN.
        """
        unique_years = sorted(set(range(start_datetime.year, end_datetime.year + 1)))

        # Create tasks to download all years in parallel
        year_tasks = [
            self._download_year_async(
                client, semaphore, station_code, parameter, year, start_datetime, end_datetime
            )
            for year in unique_years
        ]

        # Download all years in parallel
        year_results = await asyncio.gather(*year_tasks, return_exceptions=True)

        return self._process_year_results(
            unique_years, year_results, parameter, start_datetime, end_datetime
        )

    def _create_empty_dataframe(
        self, parameter: str, start_datetime: datetime, end_datetime: datetime
    ) -> pd.DataFrame:
        """
        Create an empty DataFrame with consistent structure when no data is available.

        Generates a DataFrame with a complete hourly sequence for the specified
        range, with all parameter columns initialized with NaN.

        Args:
            parameter: Climate parameter name
            start_datetime: Start date and time of the range
            end_datetime: End date and time of the range

        Returns:
            DataFrame with:
            - "Instante" column: datetime with hourly frequency
            - Parameter column(s): NaN values according to the parameter:
              * Temperatura: "Ts"
              * PuntoRocio: "Td"
              * Humedad: "HR"
              * Viento: "dd", "ff", "VRB"
              * PresionQFE: "QFE"
              * PresionQFF: "QFF"
        """
        dates = pd.date_range(start=start_datetime, end=end_datetime, freq="h")

        empty_df = pd.DataFrame({"Instante": dates})

        if parameter == "Temperatura":
            empty_df["Ts"] = np.nan
            empty_df["dl.Ts"] = ClimateDownloadStatus.DOWNLOAD_ERROR
        elif parameter == "PuntoRocio":
            empty_df["Td"] = np.nan
            empty_df["dl.Td"] = ClimateDownloadStatus.DOWNLOAD_ERROR
        elif parameter == "Humedad":
            empty_df["HR"] = np.nan
            empty_df["dl.HR"] = ClimateDownloadStatus.DOWNLOAD_ERROR
        elif parameter == "Viento":
            empty_df["dd"] = np.nan
            empty_df["ff"] = np.nan
            empty_df["VRB"] = np.nan
            empty_df["dl.dd"] = ClimateDownloadStatus.DOWNLOAD_ERROR
            empty_df["dl.ff"] = ClimateDownloadStatus.DOWNLOAD_ERROR
            empty_df["dl.VRB"] = ClimateDownloadStatus.DOWNLOAD_ERROR
        elif parameter == "PresionQFE":
            empty_df["QFE"] = np.nan
            empty_df["dl.QFE"] = ClimateDownloadStatus.DOWNLOAD_ERROR
        elif parameter == "PresionQFF":
            empty_df["QFF"] = np.nan
            empty_df["dl.QFF"] = ClimateDownloadStatus.DOWNLOAD_ERROR

        return empty_df

    def _add_status_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add download status columns (dl.*) to a successfully parsed year DataFrame.

        For each data column (anything that is not "Instante"), adds a corresponding
        dl.{col} column:
          - ClimateDownloadStatus.OK    if the value is not NaN
          - ClimateDownloadStatus.EMPTY if the value is NaN

        Args:
            df: DataFrame returned by _process_year_data

        Returns:
            DataFrame with dl.{col} columns added for every data column
        """
        for col in [c for c in df.columns if c != "Instante"]:
            df[f"dl.{col}"] = df[col].apply(
                lambda v: ClimateDownloadStatus.OK if pd.notna(v) else ClimateDownloadStatus.EMPTY
            )
        return df
