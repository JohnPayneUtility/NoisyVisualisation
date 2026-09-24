"""Multi-objective algorithms.

This package namespace is the public API: every Hydra target spells a class as
`noisyvis.algorithms.multi_objective.<Class>`, and `noisyvis.algorithms` star-imports it, so `__all__`
is explicit. Without it the star import would also export the submodule attributes and rebind
`noisyvis.algorithms.base` (deap's `base`) to this package's `base` submodule.
"""

from .base import OptimisationAlgorithm, record_pareto_data, front_sig
from .semo import SEMO, mut_flip_one_bit
from .umda import (
    MoUMDABase,
    MoUMDA,
    MoUMDA_noDuplicates,
    MoUMDA_ParetoArchive,
    MoUMDA_KMeans,
    mo_umda_update_full,
    mo_umda_update_with_archive,
)
from .nsga2 import NSGA2

__all__ = [
    "OptimisationAlgorithm",
    "record_pareto_data",
    "front_sig",
    "SEMO",
    "mut_flip_one_bit",
    "MoUMDABase",
    "MoUMDA",
    "MoUMDA_noDuplicates",
    "MoUMDA_ParetoArchive",
    "MoUMDA_KMeans",
    "mo_umda_update_full",
    "mo_umda_update_with_archive",
    "NSGA2",
]
