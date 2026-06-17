import numpy as np
import pytest
from pydantic import ValidationError

from monte.correlations import CorrelationMatrix


def test_correlation_matrix_validates_and_roundtrips_numpy() -> None:
    matrix = [[1.0, 0.5], [0.5, 1.0]]

    corr = CorrelationMatrix(matrix=matrix)

    assert corr.matrix == matrix
    np.testing.assert_array_equal(corr.to_numpy(), np.array(matrix))
    np.testing.assert_array_equal(
        CorrelationMatrix.from_numpy(np.array(matrix)).to_numpy(), np.array(matrix)
    )


def test_correlation_matrix_from_n_corr() -> None:
    corr = CorrelationMatrix.from_n_corr(3, 0.25)

    assert corr.matrix == [
        [1.0, 0.25, 0.25],
        [0.25, 1.0, 0.25],
        [0.25, 0.25, 1.0],
    ]


def test_correlation_matrix_rejects_invalid_shape_and_values() -> None:
    with pytest.raises(ValidationError, match="Matrix cannot be empty"):
        CorrelationMatrix(matrix=[])

    with pytest.raises(ValidationError, match="Matrix must be square"):
        CorrelationMatrix(matrix=[[1.0, 0.5], [0.5, 1.0], [0.0, 0.0]])

    with pytest.raises(ValidationError, match="Diagonal element"):
        CorrelationMatrix(matrix=[[1.0, 0.5], [0.5, 0.9]])

    with pytest.raises(ValidationError, match="Matrix is not symmetric"):
        CorrelationMatrix(matrix=[[1.0, 0.5], [0.3, 1.0]])

    with pytest.raises(ValidationError, match="between -1 and 1"):
        CorrelationMatrix(matrix=[[1.0, 1.2], [1.2, 1.0]])


def test_correlation_matrix_rejects_non_psd_matrix() -> None:
    with pytest.raises(ValidationError, match="positive semidefinite"):
        CorrelationMatrix(
            matrix=[
                [1.0, 0.9, 0.9],
                [0.9, 1.0, -0.9],
                [0.9, -0.9, 1.0],
            ]
        )


def test_correlation_matrix_from_numpy_requires_2d() -> None:
    with pytest.raises(ValueError, match="2-dimensional"):
        CorrelationMatrix.from_numpy(np.array([1.0, 0.5]))
