import numpy as np

from enum import IntEnum
from dataclasses import dataclass, fields


# for later from COMSOL output
class Domain(IntEnum):
    AIR = 1
    METAL = 2
    CONCRETE = 3


@dataclass(frozen=True)
class Nodes:
    xy: np.ndarray


@dataclass(frozen=True)
class Edges:
    node_idxs: np.ndarray

    def __getitem__(self, mask) -> "Edges":
        return Edges(node_idxs=self.node_idxs[mask])


@dataclass(frozen=True)
class Elements:
    node_idxs: np.ndarray
    domains: np.ndarray

    def __getitem__(self, mask) -> "Elements":
        return Elements(node_idxs=self.node_idxs[mask], domains=self.domains[mask])
