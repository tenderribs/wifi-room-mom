import numpy as np

MESH_DIR = "mesh"


from models import Nodes, Elements, Edges


def load_meshes():
    nodes_np = np.loadtxt(f"{MESH_DIR}/nodes.txt")
    elements_np = np.loadtxt(f"{MESH_DIR}/elements.txt", dtype=int)
    domains_np = np.loadtxt(f"{MESH_DIR}/domains.txt", dtype=int)

    nodes = Nodes(nodes_np)
    elements = Elements(node_idxs=elements_np, domains=domains_np)

    return nodes, elements
