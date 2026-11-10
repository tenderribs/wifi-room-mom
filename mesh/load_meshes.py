import numpy as np
import pandas as pd

MESH_DIR = "mesh"


def load_meshes():
    nodes = pd.read_csv(
        f"{MESH_DIR}/nodes.txt",
        sep=" ",
        header=None,
        usecols=range(2),
        dtype=np.float32,
    ).to_numpy()
    bcs = pd.read_csv(
        f"{MESH_DIR}/bcs.txt",
        sep=" ",
        header=None,
        usecols=range(2),
        dtype=np.uint32,
    ).to_numpy()
    elements = pd.read_csv(
        f"{MESH_DIR}/elements.txt",
        sep=" ",
        header=None,
        usecols=range(3),
        dtype=np.uint32,
    ).to_numpy()
    domains = pd.read_csv(
        f"{MESH_DIR}/domains.txt",
        sep=" ",
        header=None,
        usecols=range(1),
        dtype=np.uint32,
    ).to_numpy()
    return nodes, elements, domains, bcs
