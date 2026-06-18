from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter

import drisk as dr


def assert_model_json_round_trip(obj: BaseModel) -> None:
    """Require a Pydantic model to survive a JSON dump/validate round trip."""
    restored = type(obj).model_validate_json(obj.model_dump_json())

    assert type(restored) is type(obj)
    assert restored.model_dump(mode="json") == obj.model_dump(mode="json")


@pytest.mark.parametrize(
    "dist",
    [
        dr.Normal.elicit(10, 20, name="normal"),
        dr.LogNormal.elicit(10, 100, name="lognormal"),
        dr.LogitNormal.elicit(0.1, 0.3, name="logitnormal"),
        dr.Exponential.elicit(2.5, name="exponential"),
        dr.Gamma.elicit(3, 2.5, name="gamma"),
        dr.Beta.elicit(alpha=2, beta=5, name="beta"),
        dr.StretchedBeta.elicit(0, 5, 10, name="stretched_beta"),
        dr.PERT.elicit(0, 5, 10, name="pert"),
        dr.Bernoulli.elicit(0.3, name="bernoulli"),
        dr.Binomial.elicit(10, 0.3, name="binomial"),
        dr.Geometric.elicit(0.3, name="geometric"),
        dr.NegativeBinomial.elicit(5, 0.3, name="negative_binomial"),
        dr.Poisson.elicit(3.0, name="poisson"),
    ],
)
def test_concrete_distribution_json_round_trip(dist: dr.Distribution) -> None:
    assert_model_json_round_trip(dist)


def test_correlation_matrix_json_round_trip() -> None:
    assert_model_json_round_trip(dr.CorrelationMatrix.from_n_corr(3, 0.25))


@pytest.mark.parametrize(
    "obj",
    [
        dr.UvMixture.elicit(
            components=(dr.Normal.elicit(0, 1), dr.Normal.elicit(10, 12)),
            weights=(0.25, 0.75),
            name="mixture",
        ),
        dr.GaussianCopula.from_distributions_and_correlation(
            [dr.Normal.elicit(0, 1), dr.LogNormal.elicit(1, 3)],
            0.5,
        ),
        dr.StudentTCopula.from_distributions_and_correlation(
            [dr.Normal.elicit(0, 1), dr.LogNormal.elicit(1, 3)],
            0.5,
            nu=5,
        ),
        dr.DTree(
            dr.DecisionNode(
                "Choose",
                {
                    "A": dr.ChanceNode("Chance", {"Win": (0.2, 10), "Lose": (0.8, 0)}),
                    "B": 1,
                },
            ),
            name="tree",
        ),
    ],
)
def test_nested_pydantic_objects_json_round_trip(obj: BaseModel) -> None:
    assert_model_json_round_trip(obj)


class DistributionHolder(BaseModel):
    value: dr.SerializableDistribution


class UvDistributionHolder(BaseModel):
    value: dr.SerializableUvDistribution


class CopulaHolder(BaseModel):
    value: dr.SerializableCopula


@pytest.mark.parametrize(
    ("holder_cls", "value"),
    [
        (DistributionHolder, dr.Normal.elicit(0, 1)),
        (UvDistributionHolder, dr.Normal.elicit(0, 1)),
        (
            CopulaHolder,
            dr.GaussianCopula.from_distributions_and_correlation(
                [dr.Normal.elicit(0, 1), dr.Normal.elicit(10, 12)],
                0.25,
            ),
        ),
    ],
)
def test_abstract_base_typed_fields_json_round_trip(
    holder_cls: type[BaseModel], value: Any
) -> None:
    holder = holder_cls(value=value)

    restored = holder_cls.model_validate_json(holder.model_dump_json())

    assert type(restored.value) is type(value)
    assert restored.model_dump(mode="json") == holder.model_dump(mode="json")


def test_decision_tree_with_distribution_and_model_outcomes_json_round_trip() -> None:
    dist = dr.Normal.elicit(90, 110, name="x")
    tree = dr.DTree(
        dr.DecisionNode(
            "Choose",
            {
                "Distribution": dist,
                "Model": dist - 10,
                "Fixed": 80,
            },
        ),
        name="tree_with_uncertainty",
    )

    assert_model_json_round_trip(tree)


def test_mc_model_json_round_trip_with_pydantic_type_adapter() -> None:
    model = (dr.Normal.elicit(0, 1, name="x") + 10) * dr.LogNormal.elicit(
        1, 3, name="y"
    )

    adapter = TypeAdapter(type(model))
    restored = adapter.validate_json(adapter.dump_json(model))

    assert type(restored) is type(model)
    assert adapter.dump_python(restored, mode="json") == adapter.dump_python(
        model, mode="json"
    )
