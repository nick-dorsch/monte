"""Univariate continuous distribution interfaces and implementations."""

from .base import (
    UvBoundedContinuous,
    UvContinuous,
    UvPositiveContinuous,
    UvRealContinuous,
    UvUnitBoundedContinuous,
)


class _DistributionPendingImplementation:
    """Placeholder for distributions whose implementations are not yet available."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "This distribution has not been implemented yet."
        )


class Normal(_DistributionPendingImplementation):
    pass


class LogNormal(_DistributionPendingImplementation):
    pass


class LogitNormal(_DistributionPendingImplementation):
    pass


class Beta(_DistributionPendingImplementation):
    pass


class PERT(_DistributionPendingImplementation):
    pass


__all__ = [
    "Beta",
    "LogitNormal",
    "LogNormal",
    "Normal",
    "PERT",
    "UvBoundedContinuous",
    "UvContinuous",
    "UvPositiveContinuous",
    "UvRealContinuous",
    "UvUnitBoundedContinuous",
]
