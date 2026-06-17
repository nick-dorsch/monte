import numpy as np
import pytest

from monte.distributions import (
    Exponential,
    Gamma,
    LogitNormal,
    LogNormal,
    Normal,
    UvPositiveContinuous,
    UvRealContinuous,
    UvUnitBoundedContinuous,
)


def test_normal_elicit_sample_and_fit() -> None:
    dist = Normal.elicit(lower=10, upper=20, confidence=0.8)

    assert isinstance(dist, UvRealContinuous)
    assert dist.elicitation_params == {"lower": 10, "upper": 20, "confidence": 0.8}
    assert dist.support == (-np.inf, np.inf)
    assert dist.sample(size=4, seed=1).shape == (4,)
    np.testing.assert_array_equal(dist.rvs(size=4, seed=1), dist.sample(size=4, seed=1))

    fitted = Normal.fit([1.0, 2.0, 3.0])
    assert fitted.params["mu"] == pytest.approx(2.0)


def test_lognormal_elicit_sample_and_fit() -> None:
    dist = LogNormal.elicit(lower=10, upper=100, confidence=0.8)

    assert isinstance(dist, UvPositiveContinuous)
    assert dist.support == (0.0, np.inf)
    assert np.all(dist.sample(size=10, seed=1) > 0)

    fitted = LogNormal.fit([1.0, 2.0, 4.0])
    assert fitted.params["sigma"] > 0

    with pytest.raises(ValueError, match="positive data"):
        LogNormal.fit([0.0, 1.0])


def test_exponential_elicit_sample_fit_and_functions() -> None:
    dist = Exponential.elicit(lam=2.5)

    assert isinstance(dist, UvPositiveContinuous)
    assert dist.dist_type == "exponential"
    assert dist.params == {"lam": 2.5}
    assert dist.elicitation_params == {"lam": 2.5}
    assert dist.support == (0.0, np.inf)
    assert dist.mean == pytest.approx(0.4)
    assert dist.variance == pytest.approx(0.16)

    samples = dist.sample(size=10, seed=1)
    assert samples.shape == (10,)
    assert np.all(samples >= 0)
    np.testing.assert_array_equal(dist.rvs(size=10, seed=1), samples)

    q = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(dist.cdf(dist.ppf(q)), q)
    assert dist.pdf(0) == pytest.approx(2.5)
    assert dist.x_range[1] > dist.x_range[0] > 0

    fitted = Exponential.fit([1.0, 2.0, 3.0, 1.5, 2.5])
    assert fitted.params["lam"] > 0

    with pytest.raises(ValueError, match="positive data"):
        Exponential.fit([0.0, 1.0])


def test_exponential_validation() -> None:
    with pytest.raises(ValueError, match="lam must be positive"):
        Exponential.elicit(lam=0)
    with pytest.raises(ValueError, match="lam must be positive"):
        Exponential(params={"lam": 0})


def test_gamma_elicit_sample_fit_and_functions() -> None:
    dist = Gamma.elicit(k=3, rate=2.5)

    assert isinstance(dist, UvPositiveContinuous)
    assert dist.dist_type == "gamma"
    assert dist.params == {"alpha": 3.0, "beta": 2.5}
    assert dist.elicitation_params == {"k": 3, "rate": 2.5}
    assert dist.support == (0.0, np.inf)
    assert dist.mean == pytest.approx(1.2)
    assert dist.variance == pytest.approx(3 / 2.5**2)

    samples = dist.sample(size=10, seed=1)
    assert samples.shape == (10,)
    assert np.all(samples >= 0)
    np.testing.assert_array_equal(dist.rvs(size=10, seed=1), samples)

    q = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(dist.cdf(dist.ppf(q)), q)
    assert dist.pdf(np.array([0.1, 1.0])).shape == (2,)
    assert dist.x_range[1] > dist.x_range[0] > 0

    fitted = Gamma.fit([1.0, 2.0, 3.0, 1.5, 2.5])
    assert fitted.params["alpha"] > 0
    assert fitted.params["beta"] > 0

    with pytest.raises(ValueError, match="positive data"):
        Gamma.fit([0.0, 1.0])


def test_gamma_validation() -> None:
    with pytest.raises(ValueError, match="k.*positive"):
        Gamma.elicit(k=0, rate=1)
    with pytest.raises(ValueError, match="rate.*positive"):
        Gamma.elicit(k=1, rate=0)
    with pytest.raises(ValueError, match="alpha must be positive"):
        Gamma(params={"alpha": 0, "beta": 1})
    with pytest.raises(ValueError, match="beta must be positive"):
        Gamma(params={"alpha": 1, "beta": 0})


def test_continuous_plot_shows_cdf_with_pdf_fill() -> None:
    import matplotlib.pyplot as plt

    dist = Normal.elicit(lower=10, upper=20, confidence=0.8)
    fig, ax = plt.subplots()

    returned_ax = dist.plot(ax=ax, color="red")

    assert returned_ax is ax
    assert ax.get_ylabel() == "cumulative probability"
    assert ax.get_ylim()[0] == pytest.approx(0)
    assert ax.get_ylim()[1] == pytest.approx(1)

    assert len(fig.axes) == 2
    pdf_ax = fig.axes[1]
    assert len(pdf_ax.get_yticks()) == 0
    assert pdf_ax.get_ylabel() == ""
    assert pdf_ax.get_ylim()[0] == pytest.approx(0)

    plt.close(fig)


def test_logitnormal_elicit_sample_and_fit() -> None:
    dist = LogitNormal.elicit(lower=0.15, upper=0.25, confidence=0.8)

    assert isinstance(dist, UvUnitBoundedContinuous)
    assert dist.support == (0.0, 1.0)
    samples = dist.sample(size=10, seed=1)
    assert np.all((samples > 0) & (samples < 1))
    assert dist.pdf(np.array([-1.0, 0.5, 2.0])).tolist()[0] == 0.0
    assert dist.cdf(np.array([-1.0, 0.5, 2.0])).tolist()[2] == 1.0

    fitted = LogitNormal.fit([0.2, 0.4, 0.6])
    assert fitted.params["sigma"] > 0

    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        LogitNormal.fit([0.0, 0.5])
