"""Export of a database to `sdm`, the structured-data-models library.

This package needs the `sdm` extra: `pip install "tusk-ml[sdm]"`.
[`tusk`][tusk] does not import this package. Import it by name.
"""

from __future__ import annotations

from tusk.sdm._export import to_relational_data, to_relational_sampler

__all__ = ["to_relational_data", "to_relational_sampler"]
