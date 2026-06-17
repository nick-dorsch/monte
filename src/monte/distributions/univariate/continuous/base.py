"""Univariate continuous distribution base classes."""

from abc import ABC

from monte.distributions.univariate.base import UvDistribution


class UvContinuous(UvDistribution, ABC):
    """Base class for univariate continuous distributions."""

    pass


class UvRealContinuous(UvContinuous, ABC):
    """Continuous distribution with support over all real numbers."""

    pass


class UvPositiveContinuous(UvContinuous, ABC):
    """Continuous distribution with positive support."""

    pass


class UvBoundedContinuous(UvContinuous, ABC):
    """Continuous distribution with finite lower and upper support."""

    pass


class UvUnitBoundedContinuous(UvBoundedContinuous, ABC):
    """Continuous distribution with support on the unit interval."""

    pass
