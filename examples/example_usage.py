#!/usr/bin/env python3
"""Script de ejemplo para probar la librería atmchile."""

import asyncio
from datetime import datetime

from atmchile import ChileAirQuality, ChileClimateData

START = datetime(2024, 1, 1, 0, 0, 0)
END = datetime(2024, 1, 7, 23, 0, 0)


# ── ChileClimateData ─────────────────────────────────────────────────────────


def execute_ccd_sync() -> None:
    ccd = ChileClimateData()
    df = ccd.get_data(
        stations=["RM"],
        parameters=["Temperatura", "Humedad", "Viento"],
        start=START,
        end=END,
        region=True,
    )
    print("CCD sync:\n", df.head())


async def execute_ccd_async() -> None:
    ccd = ChileClimateData(max_concurrent_downloads=5)
    df = await ccd.get_data_async(
        stations=["RM"],
        parameters=["Temperatura", "Humedad", "Viento"],
        start=START,
        end=END,
        region=True,
    )
    print("CCD async:\n", df.head())


# ── ChileAirQuality ──────────────────────────────────────────────────────────


def execute_caq_sync() -> None:
    caq = ChileAirQuality()
    df = caq.get_data(
        stations=["RM"],
        parameters=["PM10", "PM25", "temp", "RH"],
        start=START,
        end=END,
        region=True,
    )

    print(caq.get_stations())
    print("CAQ sync:\n", df.head())


async def execute_caq_async() -> None:
    caq = ChileAirQuality(max_concurrent_downloads=5)
    df = await caq.get_data_async(
        stations=["RM"],
        parameters=["PM10", "PM25", "temp", "RH"],
        start=START,
        end=END,
        region=True,
    )
    print("CAQ async:\n", df.head())


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    execute_ccd_sync()
    asyncio.run(execute_ccd_async())
    execute_caq_sync()
    asyncio.run(execute_caq_async())
