"""Unservable forecast contracts are explicit errors, never silent resampling."""


class ForecastError(ValueError):
    """Invalid forecast inputs or incompatible metadata."""


class ForecastQueryError(ForecastError):
    """A retained draw cannot answer the requested grid or target."""


class ForecastProjectionError(ForecastError):
    """A generic result cannot establish a financial trajectory."""
