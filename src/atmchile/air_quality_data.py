from __future__ import annotations

import asyncio
import io
import os
import sys
from datetime import datetime

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """Backport of StrEnum for Python < 3.11."""


import httpx
import numpy as np
import pandas as pd
import requests

from atmchile.utils import convert_str_to_list, load_package_csv


class AirQualityDownloadStatus(StrEnum):
    """Status of a parameter measurement for a given datetime row."""

    OK = "ok"
    EMPTY = "empty"
    DOWNLOAD_ERROR = "download_error"
    CURATED = "curated"


class ChileAirQuality:
    """
    Class to obtain air quality data from SINCA monitoring stations in Chile.

    This class allows downloading and processing air quality data from different
    stations of the National Air Quality Information System (SINCA),
    with support for multiple parameters and custom date ranges.

    Available parameters (hourly data from SINCA):

    Pollutants:
        - ``PM10`` — Particulate matter ≤ 10 µm          [µg/m³N]
        - ``PM25`` — Particulate matter ≤ 2.5 µm         [µg/m³N]
        - ``SO2``  — Sulfur dioxide                       [µg/m³N]
        - ``NOX``  — Nitrogen oxides (NO + NO₂)           [ppb]
        - ``NO2``  — Nitrogen dioxide                     [ppb]
        - ``NO``   — Nitric oxide                         [ppb]
        - ``O3``   — Tropospheric ozone                   [ppb]
        - ``CO``   — Carbon monoxide                      [ppb]

    Meteorological:
        - ``temp`` — Air temperature                      [°C]
        - ``RH``   — Relative humidity                    [%]
        - ``ws``   — Wind speed                           [m/s]
        - ``wd``   — Wind direction (meteorological conv.) [°]

    For a complete list of stations and their available parameters, see the
    `Available Parameters by Station <https://github.com/maigonzalezh/atmchile-py#available-parameters-by-station>`_
    section in the repository README.md.

    Examples:
        >>> from datetime import datetime
        >>> caq = ChileAirQuality()
        >>> # Get station information
        >>> stations = caq.get_stations()
        >>>
        >>> # Download data
        >>> data = caq.get_data(
        ...     stations="El Bosque",
        ...     parameters=["PM10", "PM25"],
        ...     start=datetime(2020, 1, 1),
        ...     end=datetime(2020, 1, 2),
        ... )
        >>>
        >>> # Using station codes
        >>> data = caq.get_data(
        ...     stations=["EB", "SA"],
        ...     parameters="PM10",
        ...     start=datetime(2020, 1, 1),
        ...     end=datetime(2020, 3, 1),
        ...     site=True
        ... )
    """

    DATE_INPUT_FORMAT: str = "%d/%m/%Y"
    SINCA_DATE_FORMAT: str = "%d/%m/%Y %H:%M"

    METADATA_COLS: frozenset[str] = frozenset(
        {"date", "city", "station_code", "station_name", "region"}
    )

    # Mapping of parameters to their URL codes in SINCA
    PARAMETER_CODES: dict[str, str | dict[str, str]] = {
        "PM10": "/Cal/PM10//PM10.horario.horario.ic&",
        "PM25": "/Cal/PM25//PM25.horario.horario.ic&",
        "CO": "/Cal/0004//0004.horario.horario.ic&",
        "SO2": "/Cal/0001//0001.horario.horario.ic&",
        "NOX": "/Cal/0NOX//0NOX.horario.horario.ic&",
        "NO2": "/Cal/0003//0003.horario.horario.ic&",
        "NO": "/Cal/0002//0002.horario.horario.ic&",
        "O3": "/Cal/0008//0008.horario.horario.ic&",
        "temp": {
            "primary": "/Met/TEMP//horario_000.ic&",
            "alternative": "/Met/TEMP//horario_010.ic&",
        },
        "RH": {
            "primary": "/Met/RHUM//horario_000.ic&",
            "alternative": "/Met/RHUM//horario_002.ic&",
        },
        "ws": {
            "primary": "/Met/WSPD//horario_000.ic&",
            "alternative": "/Met/WSPD//horario_010.ic&",
        },
        "wd": {
            "primary": "/Met/WDIR//horario_000_spec.ic&",
            "alternative": "/Met/WDIR//horario_010_spec.ic&",
        },
    }

    url_base_1: str
    url_base_2: str
    max_concurrent_downloads: int
    stations_table: pd.DataFrame | None

    def __init__(
        self, stations_csv_path: str | None = None, max_concurrent_downloads: int = 5
    ) -> None:
        """
        Initialize the class with the SINCA stations table.

        Args:
            stations_csv_path: Path to the CSV file with SINCA station information.
                              If None, the packaged table is used or can be
                              set manually using the set_stations_table() method.
            max_concurrent_downloads: Maximum number of simultaneous downloads for
                                     asynchronous operations (default: 5).
        """
        self.url_base_1 = (
            "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?outtype=xcl&macro=./"
        )
        self.url_base_2 = "&path=/usr/airviro/data/CONAMA/&lang=esp&rsrc=&macropath="
        self.max_concurrent_downloads = max_concurrent_downloads

        if stations_csv_path:
            if os.path.exists(stations_csv_path):
                self.stations_table = pd.read_csv(stations_csv_path, encoding="iso-8859-1")
            else:
                self.stations_table = None
        else:
            self.stations_table = load_package_csv("sinca_stations.csv", encoding="utf-8")

    def set_stations_table(self, dataframe: pd.DataFrame) -> None:
        """
        Manually set the stations table.

        Args:
            dataframe: DataFrame with SINCA station information. Must contain
                      appropriate columns according to the SINCA CSV.
        """
        self.stations_table = dataframe

    def get_stations(self) -> pd.DataFrame:
        """
        Return the table with information of all available stations.

        Returns:
            DataFrame with SINCA station information

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
            stations: Station code(s) or region code(s)
            parameters: Air quality parameter(s) to query
            start: Start date (datetime object)
            end: End date (datetime object)

        Returns:
            Tuple with (stations_list, parameters_list, start_datetime, end_datetime)

        Raises:
            ValueError: If the stations table has not been loaded or if dates
                       are invalid.
        """
        if self.stations_table is None:
            raise ValueError("Stations table has not been loaded")

        stations_list = convert_str_to_list(stations)
        parameters_list = convert_str_to_list(parameters)

        # Expand "all" if requested
        if any(p.lower() == "all" for p in parameters_list):
            parameters_list = list(self.PARAMETER_CODES.keys())

        if any(s.lower() == "all" for s in stations_list):
            stations_list = self.stations_table["station_code"].tolist()

        if end < start:
            raise ValueError("Start date must be before end date")

        # Validate available parameters
        for param in parameters_list:
            if param not in self.PARAMETER_CODES:
                raise ValueError(
                    f"Parameter '{param}' is not available. "
                    f"Available parameters: {list(self.PARAMETER_CODES.keys())}"
                )

        return stations_list, parameters_list, start, end

    def get_data(
        self,
        stations: str | list[str],
        parameters: str | list[str],
        start: datetime,
        end: datetime,
        region: bool = False,
        curate: bool = True,
        st: bool = False,
    ) -> pd.DataFrame:
        """
        Get air quality data from the specified stations.

        Args:
            stations: Station code(s) or region code(s).
                     Use "all" for all stations.
            parameters: Air quality parameter(s) to query.
                       Use "all" for all available parameters.
            start: Start date (datetime object)
            end: End date (datetime object)
            region: If True, allows entering the region code
                   instead of the station code
            curate: If True, enables data curation for particulate matter,
                  nitrogen oxides, relative humidity and wind direction
            st: If True, includes SINCA validation reports
               ("NV": Not validated, "PV": Pre-validated, "V": Validated)

        Returns:
            DataFrame with the requested air quality data

        Examples:
            >>> from datetime import datetime
            >>> caq = ChileAirQuality()
            >>> # By station code
            >>> data = caq.get_data(
            ...     stations="RM/D14",
            ...     parameters=["PM10", "PM25"],
            ...     start=datetime(2020, 1, 1),
            ...     end=datetime(2020, 1, 2),
            ... )
            >>>
            >>> # By region code
            >>> data = caq.get_data(
            ...     stations="RM",
            ...     parameters="PM10",
            ...     start=datetime(2020, 1, 1),
            ...     end=datetime(2020, 3, 1),
            ...     region=True
            ... )

        Raises:
            ValueError: If the stations table has not been loaded or if dates
                       are invalid.
        """
        stations_list, parameters_list, start_datetime, end_datetime = (
            self._validate_and_prepare_request(stations, parameters, start, end)
        )

        dates = pd.date_range(
            start=start_datetime.replace(hour=1, minute=0, second=0),
            end=end_datetime.replace(hour=23, minute=0, second=0),
            freq="h",
        )
        dates_str = dates.strftime(self.SINCA_DATE_FORMAT)

        search_column = "region" if region else "station_code"

        dataframes_list = []

        for station in stations_list:
            try:
                matching_stations = self.stations_table[
                    self.stations_table[search_column].astype(str) == str(station)
                ]

                for _, station_row in matching_stations.iterrows():
                    try:
                        station_code = station_row["station_code"]
                        station_city = station_row["city"]
                        station_name = station_row["station_name"]
                        station_region = station_row["region"]

                        station_data = self._create_station_dataframe(
                            station_city, station_code, station_name, station_region, dates_str
                        )

                        station_data = self._combine_parameters_for_station(
                            station_data,
                            parameters_list,
                            station_code,
                            station,
                            start_datetime,
                            end_datetime,
                            st,
                        )

                        dataframes_list.append(station_data)

                    except Exception as e:
                        print(f"Error processing station {station}: {e}")
                        continue

            except Exception as e:
                print(f"Error searching for station {station}: {e}")
                continue

        if dataframes_list:
            total_data = pd.concat(dataframes_list, ignore_index=True)
        else:
            total_data = pd.DataFrame()
        if not total_data.empty:
            total_data = total_data.dropna(subset=["date", "city"])

            if curate:
                total_data = self._curate_data(total_data)

            total_data = self._convert_numeric_columns(total_data)
            total_data["date"] = pd.to_datetime(total_data["date"], format=self.SINCA_DATE_FORMAT)

        print("Data Captured!")
        return total_data

    async def get_data_async(
        self,
        stations: str | list[str],
        parameters: str | list[str],
        start: datetime,
        end: datetime,
        region: bool = False,
        curate: bool = True,
        st: bool = False,
    ) -> pd.DataFrame:
        """
        Get air quality data from the specified stations asynchronously.

        Asynchronous version that allows parallel downloads to improve performance
        when querying multiple parameters or stations.

        Args:
            stations: Station code(s) or region code(s).
                     Use "all" for all stations.
            parameters: Air quality parameter(s) to query.
                       Use "all" for all available parameters.
            start: Start date (datetime object)
            end: End date (datetime object)
            region: If True, allows entering the region code
                   instead of the station code
            curate: If True, enables data curation for particulate matter,
                  nitrogen oxides, relative humidity and wind direction
            st: If True, includes SINCA validation reports
               ("NV": Not validated, "PV": Pre-validated, "V": Validated)

        Returns:
            DataFrame with the requested air quality data

        Examples:
            >>> import asyncio
            >>> from datetime import datetime
            >>> caq = ChileAirQuality()
            >>> # By station code
            >>> async def main():
            ...     data = await caq.get_data_async(
            ...         stations="RM/D14",
            ...         parameters=["PM10", "PM25"],
            ...         start=datetime(2020, 1, 1),
            ...         end=datetime(2020, 1, 2),
            ...     )
            ...     return data
            >>> result = asyncio.run(main())
            >>>
            >>> # By region code
            >>> async def main2():
            ...     data = await caq.get_data_async(
            ...         stations="RM",
            ...         parameters="PM10",
            ...         start=datetime(2020, 1, 1),
            ...         end=datetime(2020, 3, 1),
            ...         region=True
            ...     )
            ...     return data
            >>> result = asyncio.run(main2())

        Raises:
            ValueError: If the stations table has not been loaded or if dates
                       are invalid.
        """
        stations_list, parameters_list, start_datetime, end_datetime = (
            self._validate_and_prepare_request(stations, parameters, start, end)
        )

        dates = pd.date_range(
            start=start_datetime.replace(hour=1, minute=0, second=0),
            end=end_datetime.replace(hour=23, minute=0, second=0),
            freq="h",
        )
        dates_str = dates.strftime(self.SINCA_DATE_FORMAT)

        search_column = "region" if region else "station_code"

        dataframes_list = []

        for station in stations_list:
            try:
                matching_stations = self.stations_table[
                    self.stations_table[search_column].astype(str) == str(station)
                ]

                for _, station_row in matching_stations.iterrows():
                    try:
                        station_code = station_row["station_code"]
                        station_city = station_row["city"]
                        station_name = station_row["station_name"]
                        station_region = station_row["region"]

                        station_data = self._create_station_dataframe(
                            station_city, station_code, station_name, station_region, dates_str
                        )

                        station_data = await self._combine_parameters_for_station_async(
                            station_data,
                            parameters_list,
                            station_code,
                            station,
                            start_datetime,
                            end_datetime,
                            st,
                        )

                        dataframes_list.append(station_data)

                    except Exception as e:
                        print(f"Error processing station {station}: {e}")
                        continue

            except Exception as e:
                print(f"Error searching for station {station}: {e}")
                continue

        if dataframes_list:
            total_data = pd.concat(dataframes_list, ignore_index=True)
        else:
            total_data = pd.DataFrame()
        if not total_data.empty:
            total_data = total_data.dropna(subset=["date", "city"])

            if curate:
                total_data = self._curate_data(total_data)

            total_data = self._convert_numeric_columns(total_data)
            total_data["date"] = pd.to_datetime(total_data["date"], format=self.SINCA_DATE_FORMAT)

        print("Data Captured!")
        return total_data

    def _create_station_dataframe(
        self,
        city: str,
        station_code: str,
        station_name: str,
        region: str,
        dates_str: pd.Series,
    ) -> pd.DataFrame:
        """
        Create a DataFrame with station metadata and dates.

        Args:
            city: City code of the station
            station_code: Station code
            station_name: Station name
            region: Region code
            dates_str: Series with dates formatted as strings

        Returns:
            DataFrame with station metadata columns: date, city, station_code, station_name, region
        """
        return pd.DataFrame(
            {
                "date": dates_str,
                "city": city,
                "station_code": station_code,
                "station_name": station_name,
                "region": region,
            }
        )

    def _combine_parameters_for_station(
        self,
        station_data: pd.DataFrame,
        parameters_list: list[str],
        station_code: str,
        station_name: str,
        start_datetime: datetime,
        end_datetime: datetime,
        st: bool,
    ) -> pd.DataFrame:
        """
        Combine multiple parameters into the station DataFrame.

        Args:
            station_data: Base DataFrame of the station
            parameters_list: List of parameters to download and combine
            station_code: Station code
            station_name: Station name (for messages)
            start_datetime: Start date
            end_datetime: End date
            st: Whether to include validation status

        Returns:
            DataFrame with all parameters combined
        """
        for parameter in parameters_list:
            print(f"Downloading {parameter} for {station_name}")
            param_data = self._download_parameter(
                station_code=station_code,
                parameter=parameter,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                st=st,
            )

            station_data = station_data.merge(
                param_data, left_on="date", right_on="date", how="left"
            )

        return station_data

    async def _combine_parameters_for_station_async(
        self,
        station_data: pd.DataFrame,
        parameters_list: list[str],
        station_code: str,
        station_name: str,
        start_datetime: datetime,
        end_datetime: datetime,
        st: bool,
    ) -> pd.DataFrame:
        """
        Combine multiple parameters into the station DataFrame asynchronously.

        Downloads all parameters in parallel using asyncio.gather with a semaphore
        to limit concurrency.

        Args:
            station_data: Base DataFrame of the station
            parameters_list: List of parameters to download and combine
            station_code: Station code
            station_name: Station name (for messages)
            start_datetime: Start date
            end_datetime: End date
            st: Whether to include validation status

        Returns:
            DataFrame with all parameters combined
        """
        # Create HTTP client and semaphore to limit concurrency
        async with httpx.AsyncClient(timeout=30.0) as client:
            semaphore = asyncio.Semaphore(self.max_concurrent_downloads)

            # Create tasks to download all parameters in parallel
            parameter_tasks = [
                self._download_parameter_async(
                    client,
                    semaphore,
                    station_code,
                    parameter,
                    start_datetime,
                    end_datetime,
                    st,
                )
                for parameter in parameters_list
            ]

            # Download all parameters in parallel
            parameter_results = await asyncio.gather(*parameter_tasks, return_exceptions=True)

        # Combine results into station DataFrame
        for parameter, param_data in zip(parameters_list, parameter_results, strict=True):
            if isinstance(param_data, Exception):
                print(f"Error downloading {parameter} for {station_name}: {param_data}")
                # Create empty DataFrame for this parameter
                param_data = self._create_empty_parameter_dataframe(
                    parameter, start_datetime, end_datetime
                )
            elif param_data is not None and not param_data.empty:
                station_data = station_data.merge(
                    param_data, left_on="date", right_on="date", how="left"
                )
            else:
                print(f"No data found for {parameter} for {station_name}")

        return station_data

    def _build_download_urls(
        self, station_code: str, parameter: str, start_datetime: datetime, end_datetime: datetime
    ) -> list[str]:
        """
        Build URLs to download data for a parameter.

        Args:
            station_code: Station code
            parameter: Parameter name
            start_datetime: Start date
            end_datetime: End date

        Returns:
            List of URLs to try (may have alternative URLs)
        """
        date_code = f"from={start_datetime.strftime('%y%m%d')}&to={end_datetime.strftime('%y%m%d')}"
        param_code = self.PARAMETER_CODES[parameter]

        urls_to_try = []
        if isinstance(param_code, dict):
            urls_to_try.append(
                f"{self.url_base_1}{station_code}{param_code['primary']}{date_code}{self.url_base_2}"
            )
            urls_to_try.append(
                f"{self.url_base_1}{station_code}{param_code['alternative']}{date_code}{self.url_base_2}"
            )
        else:
            urls_to_try.append(
                f"{self.url_base_1}{station_code}{param_code}{date_code}{self.url_base_2}"
            )

        return urls_to_try

    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert parameter measurement columns of the DataFrame to numeric.

        Skips metadata columns (``METADATA_COLS``), validation status columns
        (``s.*``) and download lineage columns (``dl.*``).

        Args:
            df: DataFrame to process

        Returns:
            DataFrame with numeric columns converted
        """
        if df.empty:
            return df

        numeric_cols = [
            col
            for col in df.columns
            if col not in self.METADATA_COLS and not col.startswith(("dl.", "s."))
        ]
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        return df

    def _process_validation_status(self, df: pd.DataFrame, parameter: str) -> pd.Series:
        """
        Process the validation status of the data.

        Args:
            df: DataFrame with validation columns
            parameter: Parameter name

        Returns:
            Series with validation statuses (V, PV, NV, or "")
        """
        validated = df.get("Registros validados", pd.Series(dtype=object))
        preliminary = df.get("Registros preliminares", pd.Series(dtype=object))
        not_validated = df.get("Registros no validados", pd.Series(dtype=object))

        validated_str = validated.astype(str).str.strip()
        preliminary_str = preliminary.astype(str).str.strip()
        not_validated_str = not_validated.astype(str).str.strip()

        mask_validated = pd.notna(validated) & (validated_str != "")
        mask_preliminary = ~mask_validated & pd.notna(preliminary) & (preliminary_str != "")
        mask_not_validated = (
            ~mask_validated
            & ~mask_preliminary
            & pd.notna(not_validated)
            & (not_validated_str != "")
        )

        statuses = pd.Series("", index=df.index, dtype=str)
        statuses[mask_validated] = "V"
        statuses[mask_preliminary] = "PV"
        statuses[mask_not_validated] = "NV"

        return statuses

    def _parse_sinca_dates(self, df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
        """
        Parse dates from SINCA format (YYMMDD and HHMM) to standard format.

        Args:
            df: DataFrame with SINCA data
            date_col: Name of the date column
            time_col: Name of the time column

        Returns:
            Series with dates formatted as strings
        """
        try:
            date_series = df[date_col].astype(int).astype(str).str.zfill(6)
            time_series = df[time_col].astype(int).astype(str).str.zfill(4)

            year = "20" + date_series.str[:2]
            month = date_series.str[2:4]
            day = date_series.str[4:6]
            hour = time_series.str[:2]
            minute = time_series.str[2:4]

            dates = day + "/" + month + "/" + year + " " + hour + ":" + minute
            return dates.fillna("").astype(str)
        except Exception as e:
            print(f"  Error processing date/time: {e}")
            return pd.Series([""] * len(df), dtype=str)

    def _download_parameter(
        self,
        station_code: str,
        parameter: str,
        start_datetime: datetime,
        end_datetime: datetime,
        st: bool = False,
    ) -> pd.DataFrame:
        """
        Download data for a specific parameter.

        Args:
            station_code: Station code
            parameter: Parameter name to download
            start_datetime: Start date and time
            end_datetime: End date and time
            st: If True, includes validation status column

        Returns:
            DataFrame with parameter data; always returns a DataFrame (never None).
            On download failure all rows carry the status
            ``dl.{parameter} == AirQualityDownloadStatus.DOWNLOAD_ERROR``.
        """
        urls_to_try = self._build_download_urls(
            station_code, parameter, start_datetime, end_datetime
        )

        for url in urls_to_try:
            try:
                print(f"  URL: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                df = pd.read_csv(
                    io.StringIO(response.content.decode("utf-8")),
                    sep=";",
                    decimal=",",
                    na_values="",
                )

                if df.empty or len(df.columns) < 2:
                    continue  # Try alternative URL if it exists

                return self._process_parameter_data(df, parameter, st)

            except Exception as e:
                print(f"  Error: {e}")
                if url == urls_to_try[-1]:  # Last URL
                    return self._create_empty_parameter_dataframe(
                        parameter, start_datetime, end_datetime
                    )
                continue

        return self._create_empty_parameter_dataframe(parameter, start_datetime, end_datetime)

    async def _download_url_async(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        url: str,
    ) -> bytes | None:
        """
        Download a specific URL asynchronously.

        Args:
            client: Async HTTP client from httpx
            semaphore: Semaphore to limit concurrency
            url: URL to download

        Returns:
            Response content as bytes or None if there is an error
        """
        async with semaphore:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
            except Exception as e:
                print(f"  Error downloading URL: {e}")
                return None

    async def _download_parameter_async(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        station_code: str,
        parameter: str,
        start_datetime: datetime,
        end_datetime: datetime,
        st: bool = False,
    ) -> pd.DataFrame:
        """
        Download data for a specific parameter asynchronously.

        Args:
            client: Async HTTP client from httpx
            semaphore: Semaphore to limit concurrency
            station_code: Station code
            parameter: Parameter name to download
            start_datetime: Start date and time
            end_datetime: End date and time
            st: If True, includes validation status column

        Returns:
            DataFrame with parameter data or empty DataFrame if there is an error
        """
        urls_to_try = self._build_download_urls(
            station_code, parameter, start_datetime, end_datetime
        )

        for url in urls_to_try:
            content = await self._download_url_async(client, semaphore, url)
            if content is None:
                if url == urls_to_try[-1]:  # Last URL
                    return self._create_empty_parameter_dataframe(
                        parameter, start_datetime, end_datetime
                    )
                continue

            try:
                df = pd.read_csv(
                    io.StringIO(content.decode("utf-8")),
                    sep=";",
                    decimal=",",
                    na_values="",
                )

                if df.empty or len(df.columns) < 2:
                    if url == urls_to_try[-1]:  # Last URL
                        return self._create_empty_parameter_dataframe(
                            parameter, start_datetime, end_datetime
                        )
                    continue  # Try alternative URL if it exists

                return self._process_parameter_data(df, parameter, st)
            except Exception as e:
                print(f"  Error processing data: {e}")
                if url == urls_to_try[-1]:  # Last URL
                    return self._create_empty_parameter_dataframe(
                        parameter, start_datetime, end_datetime
                    )
                continue

        return self._create_empty_parameter_dataframe(parameter, start_datetime, end_datetime)

    def _process_parameter_data(self, df: pd.DataFrame, parameter: str, st: bool) -> pd.DataFrame:
        """
        Process downloaded data according to parameter type.

        Expected SINCA CSV structure (2024+):
        - FECHA (YYMMDD): Date in YYMMDD format (e.g., 200101)
        - HORA (HHMM): Time in HHMM format (e.g., 100, 200, 2300)
        - Registros validados: Validated data
        - Registros preliminares: Preliminary data
        - Registros no validados: Non-validated data

        Args:
            df: DataFrame downloaded from SINCA
            parameter: Parameter name
            st: Whether to include validation status column

        Returns:
            Processed DataFrame with appropriate columns
        """
        date_col = "FECHA (YYMMDD)" if "FECHA (YYMMDD)" in df.columns else "FECHA"
        time_col = "HORA (HHMM)" if "HORA (HHMM)" in df.columns else "HORA"

        dates = self._parse_sinca_dates(df, date_col, time_col)

        if parameter in ["temp", "RH", "ws", "wd"]:
            data_cols = [col for col in df.columns if col not in [date_col, time_col]]

            if data_cols:
                values = df[data_cols[0]].fillna("").astype(str).str.strip()
            else:
                values = pd.Series([""] * len(df), dtype=str)

            result_df = pd.DataFrame({"date": dates, parameter: values})
        else:
            validation_cols = [
                "Registros validados",
                "Registros preliminares",
                "Registros no validados",
            ]

            existing_validation_cols = [col for col in validation_cols if col in df.columns]

            if existing_validation_cols:
                # Coalescence: take the first non-null value in priority order
                # (validated > preliminary > non-validated)
                values = df[existing_validation_cols].bfill(axis=1).iloc[:, 0]
                values = values.fillna("").astype(str).str.strip()

                result_df = pd.DataFrame({"date": dates, parameter: values})

                if st:
                    status_col = f"s.{parameter}"
                    statuses = self._process_validation_status(df, parameter)
                    result_df[status_col] = statuses.values
            else:
                result_df = pd.DataFrame({"date": dates, parameter: ""})

        result_df = result_df.fillna("")

        dl_col = f"dl.{parameter}"
        result_df[dl_col] = result_df[parameter].apply(
            lambda v: AirQualityDownloadStatus.OK
            if str(v).strip() != ""
            else AirQualityDownloadStatus.EMPTY
        )

        return result_df

    def _create_empty_parameter_dataframe(
        self, parameter: str, start_datetime: datetime, end_datetime: datetime
    ) -> pd.DataFrame:
        """
        Create an empty DataFrame for a parameter when no data is available.

        Args:
            parameter: Parameter name
            start_datetime: Start date
            end_datetime: End date

        Returns:
            DataFrame with correct structure but empty values
        """
        dates = pd.date_range(
            start=start_datetime.replace(hour=1, minute=0, second=0),
            end=end_datetime.replace(hour=23, minute=0, second=0),
            freq="h",
        )
        dates_str = dates.strftime(self.SINCA_DATE_FORMAT)

        empty_df = pd.DataFrame({"date": dates_str, parameter: ""})
        empty_df[f"dl.{parameter}"] = AirQualityDownloadStatus.DOWNLOAD_ERROR

        return empty_df

    def _curate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data curation according to quality rules.

        Implemented rules:
        1. NOX: NO + NO2 must be <= NOX * 1.001
        2. PM: PM2.5 must be <= PM10 * 1.001
        3. Wind direction: 0 <= wd <= 360
        4. Relative humidity: 0 <= RH <= 100

        Args:
            df: DataFrame with data to curate

        Returns:
            DataFrame with curated data
        """
        df_curated = df.copy()

        def _mark_curated(mask: pd.Series, params: list[str]) -> None:
            """Mark dl.{param} as CURATED for rows where was previously OK, then nullify."""
            for param in params:
                dl_col = f"dl.{param}"
                if dl_col in df_curated.columns:
                    was_ok = df_curated[dl_col] == AirQualityDownloadStatus.OK
                    df_curated.loc[mask & was_ok, dl_col] = AirQualityDownloadStatus.CURATED
            df_curated.loc[mask, params] = np.nan

        # Nitrogen oxides
        if all(col in df_curated.columns for col in ["NO", "NO2", "NOX"]):
            try:
                mask = (
                    pd.to_numeric(df_curated["NO"], errors="coerce")
                    + pd.to_numeric(df_curated["NO2"], errors="coerce")
                ) > (pd.to_numeric(df_curated["NOX"], errors="coerce") * 1.001)

                _mark_curated(mask, ["NO", "NO2", "NOX"])
                print("  NOX curation applied")
            except Exception as e:
                print(f"  Error in NOX curation: {e}")

        # Particulate matter
        if all(col in df_curated.columns for col in ["PM25", "PM10"]):
            try:
                mask = pd.to_numeric(df_curated["PM25"], errors="coerce") > (
                    pd.to_numeric(df_curated["PM10"], errors="coerce") * 1.001
                )

                _mark_curated(mask, ["PM10", "PM25"])
                print("  PM curation applied")
            except Exception as e:
                print(f"  Error in PM curation: {e}")

        # Wind direction
        if "wd" in df_curated.columns:
            try:
                wd_numeric = pd.to_numeric(df_curated["wd"], errors="coerce")
                mask = (wd_numeric > 360) | (wd_numeric < 0)
                _mark_curated(mask, ["wd"])
                print("  Wind direction curation applied")
            except Exception as e:
                print(f"  Error in wd curation: {e}")

        # Relative humidity
        if "RH" in df_curated.columns:
            try:
                rh_numeric = pd.to_numeric(df_curated["RH"], errors="coerce")
                mask = (rh_numeric > 100) | (rh_numeric < 0)
                _mark_curated(mask, ["RH"])
                print("  Relative humidity curation applied")
            except Exception as e:
                print(f"  Error in RH curation: {e}")

        return df_curated
