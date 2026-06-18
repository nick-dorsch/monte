"""Serializable copula type aliases."""

from typing import Annotated

from pydantic import Field

from drisk.copulas.gaussian import GaussianCopula
from drisk.copulas.student_t import StudentTCopula

SerializableCopula = Annotated[
    GaussianCopula | StudentTCopula,
    Field(discriminator="copula_type"),
]
