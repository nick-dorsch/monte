import monte
from monte import (
    Copula,
    CorrelationMatrix,
    Exponential,
    Gamma,
    GaussianCopula,
    StudentTCopula,
)
from monte.distributions import Distribution, UvContinuous, UvDistribution


def test_package_imports() -> None:
    assert monte.__name__ == "monte"
    assert Distribution is not None
    assert Exponential is monte.Exponential
    assert UvDistribution is not None
    assert UvContinuous is not None
    assert Copula is not None
    assert CorrelationMatrix is not None
    assert GaussianCopula is not None
    assert Gamma is monte.Gamma
    assert StudentTCopula is not None
