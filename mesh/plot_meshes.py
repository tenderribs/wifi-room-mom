import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from models import Nodes, Elements, Domain


def plot_meshes(nodes: Nodes, elements: Elements) -> None:
    xy, tris, doms = nodes.xy, elements.node_idxs, elements.domains

    # Longest edge of each triangle = local element size h
    corners = xy[tris]  # (T, 3 corners, 2 coords)
    edge_len = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)  # (T, 3)
    h_tri = edge_len.max(axis=1)  # (T,)

    present = [Domain(dom) for dom in np.unique(doms)]
    print(f"{len(xy)} nodes, {len(tris)} triangles")
    for dom in present:
        print(f"  {dom.name:<9} {np.sum(doms == dom):>6} triangles")
    print(
        f"element size h: min {h_tri.min():.4g}, median {np.median(h_tri):.4g}, max {h_tri.max():.4g} m"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")

    # Left: triangles coloured by domain (one colour per domain present in the mesh)
    DOM_COLORS = {
        Domain.AIR: "#cfe3f5",
        Domain.METAL: "#9e9e9e",
        Domain.CONCRETE: "#e8c39e",
    }
    dom_idx = np.searchsorted([int(dom) for dom in present], doms)  # 0..len(present)-1
    axes[0].tripcolor(
        xy[:, 0],
        xy[:, 1],
        tris,
        facecolors=dom_idx,
        cmap=ListedColormap([DOM_COLORS[dom] for dom in present]),
        vmin=-0.5,
        vmax=len(present) - 0.5,
    )
    axes[0].triplot(xy[:, 0], xy[:, 1], tris, color="black", lw=0.2)
    axes[0].legend(
        handles=[
            Patch(color=DOM_COLORS[dom], label=dom.name.lower()) for dom in present
        ],
        loc="upper right",
    )
    axes[0].set_title("Mesh by domain")

    # Right: local element size, to check the h_shape -> h_far grading
    im = axes[1].tripcolor(xy[:, 0], xy[:, 1], tris, facecolors=h_tri, cmap="viridis")
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="longest edge h (m)")
    axes[1].set_title("Element size")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    plt.show()
