"""Composable Monte Carlo model expressions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from monte.arithmetic import ArithmeticMixin
from monte.distributions import Distribution
from monte.random import SeedLike, get_rng

Operation = Callable[..., Any]
Operand = Any
EvaluationCache = dict[tuple[type[Any], int], np.ndarray]


@dataclass(frozen=True)
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
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """
        Evaluate the model by sampling distribution leaves and applying operations.

        A single shared random number generator is passed through all distribution
        samples so a seeded model is reproducible while each distribution consumes
        a distinct part of the random stream. Reused distribution/model objects are
        cached within one evaluation, preserving sample-wise dependence: ``x + x``
        uses the same draws from ``x`` on both sides.
        """
        rng = get_rng(seed)
        cache: EvaluationCache = {}
        return np.asarray(self._eval(size=size, seed=rng, cache=cache))

    def rvs(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Alias for :meth:`sample` for users familiar with SciPy naming."""
        return self.sample(size=size, seed=seed)

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
