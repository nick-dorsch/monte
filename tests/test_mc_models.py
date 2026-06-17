import numpy as np

import monte as mt
from monte.models import MCModel


def test_distribution_arithmetic_returns_mc_model() -> None:
    x = mt.Normal.elicit(-1.0, 1.0)
    y = mt.Normal.elicit(9.0, 11.0)

    model = x + y * 2 - 3

    assert isinstance(model, MCModel)
    assert model.sample(size=5, seed=123).shape == (5,)


def test_model_arithmetic_matches_numpy_with_shared_rng() -> None:
    x = mt.Normal.elicit(-1.0, 1.0)
    y = mt.LogNormal.elicit(8.0, 12.0)
    model = x * y + 2

    rng = np.random.default_rng(123)
    expected = x.sample(size=10, seed=rng) * y.sample(size=10, seed=rng) + 2

    np.testing.assert_allclose(model.sample(size=10, seed=123), expected)


def test_reused_distribution_object_uses_same_samples() -> None:
    x = mt.Normal.elicit(-1.0, 1.0)

    doubled = (x + x).sample(size=10, seed=123)
    expected = (2 * x).sample(size=10, seed=123)

    np.testing.assert_allclose(doubled, expected)


def test_reverse_and_unary_operations() -> None:
    x = mt.Normal.elicit(1.0, 3.0)

    rng = np.random.default_rng(123)
    x_samples = x.sample(size=8, seed=rng)
    expected = abs(-((100 / x_samples) ** 2))

    model = abs(-((100 / x) ** 2))

    np.testing.assert_allclose(model.sample(size=8, seed=123), expected)
