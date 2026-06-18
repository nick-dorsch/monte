"""Serializable distribution type aliases."""

from typing import Annotated

from pydantic import Field

from drisk.distributions.univariate.continuous import (
    PERT,
    Beta,
    Exponential,
    Gamma,
    LogitNormal,
    LogNormal,
    Normal,
    StretchedBeta,
)
from drisk.distributions.univariate.discrete import (
    Bernoulli,
    Binomial,
    Geometric,
    NegativeBinomial,
    Poisson,
)

SerializableUvDistribution = Annotated[
    Normal
    | LogNormal
    | LogitNormal
    | Exponential
    | Gamma
    | Beta
    | StretchedBeta
    | PERT
    | Bernoulli
    | Binomial
    | Geometric
    | NegativeBinomial
    | Poisson,
    Field(discriminator="dist_type"),
]

SerializableDistribution = SerializableUvDistribution
