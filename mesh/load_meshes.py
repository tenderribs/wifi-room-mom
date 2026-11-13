import numpy as np


from models import Nodes, Elements


def load_meshes(src_dir: str = "mesh/sample"):
    nodes_np = np.loadtxt(f"{src_dir}/nodes.txt")
    elements_np = np.loadtxt(f"{src_dir}/elements.txt", dtype=int)
    domains_np = np.loadtxt(f"{src_dir}/domains.txt", dtype=int)

    nodes = Nodes(xy=nodes_np)
    elements = Elements(node_idxs=elements_np, domains=domains_np)

    return nodes, elements
