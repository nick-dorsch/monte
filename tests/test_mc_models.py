import numpy as np
import pytest

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


def test_model_summary_returns_configurable_dataframe() -> None:
    x = mt.Normal.elicit(-1.0, 1.0)
    model = x * 2

    summary = model.summary(
        size=1_000,
        seed=123,
        threshold=0,
        percentiles=(99, 90, 75, 50, 25, 10, 1),
    )

    assert summary.index.name == "metric"
    assert summary.index.tolist() == ["value"]
    assert summary.columns.tolist() == [
        "mean",
        "p(> 0)",
        "p99",
        "p90",
        "p75",
        "p50",
        "p25",
        "p10",
        "p1",
    ]
    assert summary.loc["value", "p(> 0)"] == pytest.approx(
        np.mean(model.sample(size=1_000, seed=123) > 0)
    )


def test_model_plot_shows_ecdf_with_histogram_fill() -> None:
    import matplotlib.pyplot as plt

    x = mt.Normal.elicit(-1.0, 1.0)
    model = x * 2
    fig, ax = plt.subplots()

    returned_ax = model.plot(ax=ax, size=100, seed=123, color="red")

    assert returned_ax is ax
    assert ax.get_ylabel() == "cumulative probability"
    assert ax.get_ylim()[0] == pytest.approx(0)
    assert ax.get_ylim()[1] == pytest.approx(1)

    assert len(fig.axes) == 2
    hist_ax = fig.axes[1]
    assert len(hist_ax.get_yticks()) == 0
    assert hist_ax.get_ylabel() == ""
    assert hist_ax.get_ylim()[0] == pytest.approx(0)

    plt.close(fig)


def test_reverse_and_unary_operations() -> None:
    x = mt.Normal.elicit(1.0, 3.0)

    rng = np.random.default_rng(123)
    x_samples = x.sample(size=8, seed=rng)
    expected = abs(-((100 / x_samples) ** 2))

    model = abs(-((100 / x) ** 2))

    np.testing.assert_allclose(model.sample(size=8, seed=123), expected)
