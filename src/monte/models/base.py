"""Composable Monte Carlo model expressions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from monte.arithmetic import ArithmeticMixin
from monte.distributions import Distribution
from monte.random import SeedLike, get_rng
from monte.summary import percentile_label, threshold_probability_label

Operation = Callable[..., Any]
Operand = Any
EvaluationCache = dict[tuple[type[Any], int], np.ndarray]


@dataclass
class MCModel(ArithmeticMixin):
    """
    Lazy Monte Carlo expression composed from distributions, constants, and models.

    ``MCModel`` is intentionally not a ``Distribution``: composed Monte Carlo
    expressions are sampleable, but generally do not have closed-form density,
    CDF, or inverse-CDF methods.
    """

    op: Operation
    operands: tuple[Operand, ...]
    name: str | None = None
    _cached_samples: np.ndarray | None = field(default=None, init=False, repr=False)
    _cached_size: int | tuple[int, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_seed: SeedLike = field(default=None, init=False, repr=False)

    @classmethod
    def from_operation(
        cls,
        op: Operation,
        *operands: Operand,
        name: str | None = None,
    ) -> MCModel:
        """Build a lazy model expression from an operation and operands."""
        return cls(op=op, operands=operands, name=name)

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
        refresh: bool = False,
    ) -> np.ndarray:
        """
        Evaluate the model by sampling distribution leaves and applying operations.

        A single shared random number generator is passed through all distribution
        samples so a seeded model is reproducible while each distribution consumes
        a distinct part of the random stream. Reused distribution/model objects are
        cached within one evaluation, preserving sample-wise dependence: ``x + x``
        uses the same draws from ``x`` on both sides.
        """
        return self._get_samples(size=size, seed=seed, refresh=refresh)

    def rvs(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
        refresh: bool = False,
    ) -> np.ndarray:
        """Alias for :meth:`sample` for users familiar with SciPy naming."""
        return self.sample(size=size, seed=seed, refresh=refresh)

    def clear_cache(self) -> None:
        """Clear cached simulation results from this model."""
        self._cached_samples = None
        self._cached_size = None
        self._cached_seed = None

    def _get_samples(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
        refresh: bool = False,
        default_size: int = 10_000,
    ) -> np.ndarray:
        """Return cached samples when suitable, otherwise simulate and cache them."""
        if not refresh and self._cached_samples is not None:
            size_matches = size is None or self._cached_size == size
            seed_matches = seed is None or self._cached_seed == seed
            if size_matches and seed_matches:
                return self._cached_samples

        sample_size = default_size if size is None else size
        rng = get_rng(seed)
        cache: EvaluationCache = {}
        samples = np.asarray(self._eval(size=sample_size, seed=rng, cache=cache))
        self._cached_samples = samples
        self._cached_size = sample_size
        self._cached_seed = seed
        return samples

    def summary(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
        refresh: bool = False,
        threshold: float | int | None = None,
        percentiles: list[float | int] | tuple[float | int, ...] = (90, 50, 10),
    ) -> pd.DataFrame:
        """
        Summarize simulated model outcomes as a tidy dataframe.

        The summary includes the mean, configurable percentiles, and optionally
        the probability that simulated values exceed ``threshold``. Cached
        samples are reused by default when available.
        """
        samples = np.ravel(self._get_samples(size=size, seed=seed, refresh=refresh))
        values: dict[str, float] = {"mean": float(np.mean(samples))}

        if threshold is not None:
            values[threshold_probability_label(threshold)] = float(
                np.mean(samples > threshold)
            )

        percentile_values = np.percentile(samples, percentiles)
        values.update(
            {
                percentile_label(percentile): float(value)
                for percentile, value in zip(
                    percentiles, percentile_values, strict=True
                )
            }
        )

        index_label = self.name or "value"
        return pd.DataFrame(values, index=pd.Index([index_label], name="metric"))

    def plot(
        self,
        ax: Any = None,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
        refresh: bool = False,
        bins: int | str = 80,
        show: bool = False,
        ecdf_kwargs: dict[str, Any] | None = None,
        hist_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Plot an empirical CDF with a low-alpha histogram on a secondary axis.

        Returns the primary Matplotlib ``Axes`` object. Extra keyword arguments
        are passed to the ECDF line for convenient calls like
        ``model.plot(color="steelblue")``. Cached samples are reused by default
        when available.
        """
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        samples = np.sort(
            np.ravel(self._get_samples(size=size, seed=seed, refresh=refresh))
        )
        ecdf = np.arange(1, samples.size + 1) / samples.size

        line_kwargs = {**(ecdf_kwargs or {}), **kwargs}
        (ecdf_line,) = ax.plot(samples, ecdf, **line_kwargs)

        hist_ax = ax.twinx()
        fill_kwargs = {
            "color": ecdf_line.get_color(),
            "alpha": 0.2,
            **(hist_kwargs or {}),
        }
        hist_ax.hist(samples, bins=bins, **fill_kwargs)

        ax.set_xlabel(self.name or "value")
        ax.set_ylabel("cumulative probability")
        ax.set_ylim(bottom=0, top=1)
        ax.set_title(self.name or "Monte Carlo model")

        hist_ax.set_ylim(bottom=0)
        hist_ax.set_yticks([])
        hist_ax.set_ylabel("")
        hist_ax.spines["right"].set_visible(False)

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        return ax

    def _eval(
        self,
        *,
        size: int | tuple[int, ...],
        seed: SeedLike,
        cache: EvaluationCache,
    ) -> np.ndarray:
        """Evaluate this expression node using a shared cache."""
        key = (type(self), id(self))
        if key in cache:
            return cache[key]

        values = [
            _eval_operand(operand, size=size, seed=seed, cache=cache)
            for operand in self.operands
        ]
        result = np.asarray(self.op(*values))
        cache[key] = result
        return result


def _eval_operand(
    operand: Operand,
    *,
    size: int | tuple[int, ...],
    seed: SeedLike,
    cache: EvaluationCache,
) -> np.ndarray:
    """Evaluate an operand as an array suitable for NumPy broadcasting."""
    if isinstance(operand, MCModel):
        return operand._eval(size=size, seed=seed, cache=cache)

    if isinstance(operand, Distribution):
        key = (type(operand), id(operand))
        if key not in cache:
            cache[key] = np.asarray(operand.sample(size=size, seed=seed))
        return cache[key]

    return np.asarray(operand)
