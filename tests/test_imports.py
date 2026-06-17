import monte
from monte import Copula, CorrelationMatrix, GaussianCopula, StudentTCopula
from monte.distributions import Distribution, UvContinuous, UvDistribution


def test_package_imports() -> None:
    assert monte.__name__ == "monte"
    assert Distribution is not None
    assert UvDistribution is not None
    assert UvContinuous is not None
    assert Copula is not None
    assert CorrelationMatrix is not None
    assert GaussianCopula is not None
    assert StudentTCopula is not None
