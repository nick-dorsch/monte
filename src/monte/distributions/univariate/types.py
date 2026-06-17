"""Univariate distribution type discriminators."""

from enum import StrEnum


class UvContinuousType(StrEnum):
    """Supported univariate continuous distribution types."""

    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    LOGITNORMAL = "logitnormal"
    BETA = "beta"
    PERT = "pert"
