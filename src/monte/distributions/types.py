"""Distribution type discriminators and shared typing helpers."""

from typing import Any, Protocol, Sequence

import numpy as np


class DataFrameLike(Protocol):
    """Structural type for pandas/polars-style tabular data."""

    def to_numpy(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Return the tabular data as a NumPy array."""
        ...


# Input data accepted by distribution fitting APIs.
type ArrayLike = np.ndarray | Sequence[Any] | DataFrameLike
