"""Python library to obtain climate and air quality data from monitoring stations in Chile."""

from importlib.metadata import version

__version__ = version("atmchile")

from atmchile.air_quality_data import ChileAirQuality
from atmchile.climate_data import ChileClimateData

__all__ = ["ChileClimateData", "ChileAirQuality", "__version__"]
