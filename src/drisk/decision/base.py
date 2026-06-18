"""Decision tree support for business decision analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from drisk.distributions import Distribution
from drisk.models import MCModel
from drisk.random import SeedLike, get_rng
from drisk.summary import percentile_label

ValueLike = Any


class DTreeNode(BaseModel, ABC):
    """Abstract base class for decision tree nodes."""

    node_type: str
    name: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @abstractmethod
    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        """Return the expected value of this node."""
        pass

    @abstractmethod
    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        """Generate outcome samples from this node."""
        pass

    @abstractmethod
    def rollback_rows(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> list[dict[str, Any]]:
        """Return tidy rollback rows for this node and descendants."""
        pass


class OutcomeNode(DTreeNode):
    """Terminal node containing a scalar, distribution, or Monte Carlo model value."""

    node_type: Literal["outcome"] = "outcome"
    value: ValueLike

    def __init__(
        self, value: ValueLike = 0, name: str | None = None, **data: Any
    ) -> None:
        if "value" not in data:
            data["value"] = value
        if name is not None and "name" not in data:
            data["name"] = name
        super().__init__(**data)

    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        samples = np.ravel(
            _sample_value(self.value, size=10_000 if size is None else size, seed=seed)
        )
        return float(np.mean(samples))

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        return _sample_value(self.value, size=size, seed=seed)

    def rollback_rows(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "node": self.name or "outcome",
                "node_type": self.node_type,
                "branch": None,
                "probability": None,
                "expected_value": self.expected_value(size=size, seed=seed),
                "selected": None,
            }
        ]


class DecisionBranch(BaseModel):
    """A branch leaving a decision node."""

    name: str
    node: SerializeAsAny[DTreeNode]
    value: ValueLike = 0

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def coerce_node(cls, data: Any) -> Any:
        if isinstance(data, dict) and "node" in data:
            copied = dict(data)
            copied["node"] = as_node(copied["node"])
            return copied
        return data


class ChanceBranch(BaseModel):
    """A probability-weighted branch leaving a chance node."""

    name: str
    probability: float = Field(ge=0)
    node: SerializeAsAny[DTreeNode]
    value: ValueLike = 0

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def coerce_node(cls, data: Any) -> Any:
        if isinstance(data, dict) and "node" in data:
            copied = dict(data)
            copied["node"] = as_node(copied["node"])
            return copied
        return data


class DecisionNode(DTreeNode):
    """Decision node that selects the branch with the highest expected value."""

    node_type: Literal["decision"] = "decision"
    branches: tuple[DecisionBranch, ...]

    def __init__(
        self,
        name: str | None = None,
        branches: Any = None,
        **data: Any,
    ) -> None:
        if name is not None and "name" not in data:
            data["name"] = name
        if branches is not None and "branches" not in data:
            data["branches"] = branches
        super().__init__(**data)

    @field_validator("branches", mode="before")
    @classmethod
    def coerce_branches(cls, branches: Any) -> tuple[DecisionBranch, ...]:
        return tuple(_coerce_decision_branches(branches))

    @field_validator("branches")
    @classmethod
    def validate_branches(
        cls, branches: tuple[DecisionBranch, ...]
    ) -> tuple[DecisionBranch, ...]:
        if not branches:
            raise ValueError("DecisionNode requires at least one branch")
        return branches

    def selected_branch(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> DecisionBranch:
        values = [
            _branch_expected_value(branch, size=size, seed=seed)
            for branch in self.branches
        ]
        return self.branches[int(np.argmax(values))]

    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        return _branch_expected_value(
            self.selected_branch(size=size, seed=seed), size=size, seed=seed
        )

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        branch = self.selected_branch(size=size, seed=seed)
        return _sample_branch(branch, size=size, seed=seed)

    def rollback_rows(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> list[dict[str, Any]]:
        values = [
            _branch_expected_value(branch, size=size, seed=seed)
            for branch in self.branches
        ]
        selected_index = int(np.argmax(values))
        rows = [
            {
                "node": self.name or "decision",
                "node_type": self.node_type,
                "branch": branch.name,
                "probability": None,
                "expected_value": value,
                "selected": i == selected_index,
            }
            for i, (branch, value) in enumerate(zip(self.branches, values, strict=True))
        ]
        for branch in self.branches:
            rows.extend(branch.node.rollback_rows(size=size, seed=seed))
        return rows


class ChanceNode(DTreeNode):
    """Chance node that follows branches according to their probabilities."""

    node_type: Literal["chance"] = "chance"
    branches: tuple[ChanceBranch, ...]

    def __init__(
        self,
        name: str | None = None,
        branches: Any = None,
        **data: Any,
    ) -> None:
        if name is not None and "name" not in data:
            data["name"] = name
        if branches is not None and "branches" not in data:
            data["branches"] = branches
        super().__init__(**data)

    @field_validator("branches", mode="before")
    @classmethod
    def coerce_branches(cls, branches: Any) -> tuple[ChanceBranch, ...]:
        return tuple(_coerce_chance_branches(branches))

    @field_validator("branches")
    @classmethod
    def validate_branches(
        cls, branches: tuple[ChanceBranch, ...]
    ) -> tuple[ChanceBranch, ...]:
        if not branches:
            raise ValueError("ChanceNode requires at least one branch")
        probability_sum = sum(branch.probability for branch in branches)
        if not np.isclose(probability_sum, 1.0):
            raise ValueError(
                f"ChanceNode branch probabilities must sum to 1, got {probability_sum}"
            )
        return branches

    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        return float(
            sum(
                branch.probability
                * _branch_expected_value(branch, size=size, seed=seed)
                for branch in self.branches
            )
        )

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        rng = get_rng(seed)
        probabilities = [branch.probability for branch in self.branches]
        choices = rng.choice(len(self.branches), size=size, p=probabilities)
        branch_samples = [
            _sample_branch(branch, size=size, seed=rng) for branch in self.branches
        ]
        result = np.empty(np.shape(choices), dtype=float)
        for i, samples in enumerate(branch_samples):
            result[choices == i] = np.asarray(samples, dtype=float)[choices == i]
        return result

    def rollback_rows(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> list[dict[str, Any]]:
        rows = [
            {
                "node": self.name or "chance",
                "node_type": self.node_type,
                "branch": branch.name,
                "probability": branch.probability,
                "expected_value": _branch_expected_value(branch, size=size, seed=seed),
                "selected": None,
            }
            for branch in self.branches
        ]
        for branch in self.branches:
            rows.extend(branch.node.rollback_rows(size=size, seed=seed))
        return rows


class DTree(BaseModel):
    """A sampleable, serializable decision tree."""

    root: SerializeAsAny[DTreeNode]
    name: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def __init__(self, root: Any = None, name: str | None = None, **data: Any) -> None:
        if root is not None and "root" not in data:
            data["root"] = root
        if name is not None and "name" not in data:
            data["name"] = name
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def coerce_root(cls, data: Any) -> Any:
        if isinstance(data, dict) and "root" in data:
            copied = dict(data)
            copied["root"] = as_node(copied["root"])
            return copied
        return data

    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        """Return the rolled-back expected value of the tree."""
        return self.root.expected_value(size=size, seed=seed)

    def rollback(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
        precision: int | None = 2,
    ) -> pd.DataFrame:
        """Return a tidy rollback table with selected decision branches."""
        frame = pd.DataFrame(self.root.rollback_rows(size=size, seed=seed))
        if precision is not None and "expected_value" in frame:
            frame["expected_value"] = frame["expected_value"].round(precision)
        return frame

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        """Sample outcomes under the rollback-selected policy, without perfect information."""
        return self.root.sample(size=size, seed=seed)

    def rvs(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        """Alias for :meth:`sample`."""
        return self.sample(size=size, seed=seed)

    def summary(
        self,
        *,
        size: int | tuple[int, ...] = 10_000,
        seed: SeedLike = None,
        percentiles: list[float | int] | tuple[float | int, ...] = (90, 50, 10),
        precision: int | None = 2,
    ) -> pd.DataFrame:
        """Summarize simulated outcomes under the rollback-selected policy."""
        samples = np.ravel(self.sample(size=size, seed=seed))
        values: dict[str, float] = {"mean": float(np.mean(samples))}
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
        summary = pd.DataFrame(values, index=pd.Index([index_label], name="metric"))
        if precision is not None:
            summary = summary.round(precision)
        return summary

    def plot(
        self,
        ax: Any = None,
        *,
        size: int | tuple[int, ...] = 10_000,
        seed: SeedLike = None,
        bins: int | str = 80,
        show: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot sampled tree outcomes as an empirical CDF with a histogram."""
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        samples = np.sort(np.ravel(self.sample(size=size, seed=seed)))
        ecdf = np.arange(1, samples.size + 1) / samples.size
        (ecdf_line,) = ax.plot(samples, ecdf, **kwargs)

        hist_ax = ax.twinx()
        hist_ax.hist(samples, bins=bins, color=ecdf_line.get_color(), alpha=0.2)
        hist_ax.set_yticks([])
        hist_ax.spines["right"].set_visible(False)

        ax.set_xlabel(self.name or "value")
        ax.set_ylabel("cumulative probability")
        ax.set_ylim(bottom=0, top=1)
        ax.set_title(self.name or "Decision tree outcome")

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        return ax


def as_node(value: Any) -> DTreeNode:
    """Return ``value`` as a decision tree node."""
    if isinstance(value, DTreeNode):
        return value
    if isinstance(value, dict) and "node_type" in value:
        node_type = value["node_type"]
        if node_type == "decision":
            return DecisionNode.model_validate(value)
        if node_type == "chance":
            return ChanceNode.model_validate(value)
        if node_type == "outcome":
            return OutcomeNode.model_validate(value)
        raise ValueError(f"Unknown decision tree node_type: {node_type}")
    return OutcomeNode(value=value)


def _coerce_decision_branches(branches: Any) -> list[DecisionBranch]:
    if isinstance(branches, dict):
        return [
            DecisionBranch(name=str(name), node=as_node(value))
            for name, value in branches.items()
        ]
    return [
        branch
        if isinstance(branch, DecisionBranch)
        else DecisionBranch.model_validate(branch)
        for branch in branches
    ]


def _coerce_chance_branches(branches: Any) -> list[ChanceBranch]:
    if isinstance(branches, dict):
        coerced = []
        for name, spec in branches.items():
            if isinstance(spec, tuple) and len(spec) == 2:
                probability, value = spec
                coerced.append(
                    ChanceBranch(
                        name=str(name), probability=probability, node=as_node(value)
                    )
                )
            else:
                coerced.append(ChanceBranch.model_validate({"name": name, **spec}))
        return coerced
    return [
        branch
        if isinstance(branch, ChanceBranch)
        else ChanceBranch.model_validate(branch)
        for branch in branches
    ]


def _branch_expected_value(
    branch: DecisionBranch | ChanceBranch,
    *,
    size: int | tuple[int, ...] | None = None,
    seed: SeedLike = None,
) -> float:
    rng = get_rng(seed)
    return _value_expected_value(
        branch.value, size=size, seed=rng
    ) + branch.node.expected_value(
        size=size,
        seed=rng,
    )


def _value_expected_value(
    value: ValueLike,
    *,
    size: int | tuple[int, ...] | None = None,
    seed: SeedLike = None,
) -> float:
    samples = np.ravel(
        _sample_value(value, size=10_000 if size is None else size, seed=seed)
    )
    return float(np.mean(samples))


def _sample_branch(
    branch: DecisionBranch | ChanceBranch,
    *,
    size: int | tuple[int, ...],
    seed: SeedLike = None,
) -> np.ndarray:
    rng = get_rng(seed)
    return _sample_value(branch.value, size=size, seed=rng) + branch.node.sample(
        size=size, seed=rng
    )


def _sample_value(
    value: ValueLike,
    *,
    size: int | tuple[int, ...],
    seed: SeedLike = None,
) -> np.ndarray:
    if isinstance(value, Distribution | MCModel):
        return np.asarray(value.sample(size=size, seed=seed), dtype=float)
    if isinstance(value, DTree):
        return np.asarray(value.sample(size=size, seed=seed), dtype=float)
    return np.broadcast_to(np.asarray(value, dtype=float), size)
