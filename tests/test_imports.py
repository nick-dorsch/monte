import monte
from monte.distributions import Distribution, UvContinuous, UvDistribution


def test_package_imports() -> None:
    assert monte.__name__ == "monte"
    assert Distribution is not None
    assert UvDistribution is not None
    assert UvContinuous is not None
