from dataclasses import dataclass
from pathlib import Path
from models import Domain

import gmsh
import numpy as np


@dataclass
class Stadium:
    x: float  # centre
    y: float
    length: float  # length of the straight part
    r: float  # radius of the round ends (half the thickness)
    angle: float = 0.0  # rotation about the centre [rad]
    domain: Domain = Domain.METAL


def add_stadium(s: Stadium) -> int:
    """Build a stadium from 2 lines + 2 half-circle arcs; return the surface tag."""
    occ = gmsh.model.occ
    a = s.length / 2

    # corner points of the straight segments, plus the tips of the two arcs
    p1 = occ.addPoint(s.x - a, s.y - s.r, 0)
    p2 = occ.addPoint(s.x + a, s.y - s.r, 0)
    p3 = occ.addPoint(s.x + a, s.y + s.r, 0)
    p4 = occ.addPoint(s.x - a, s.y + s.r, 0)

    tip_right = occ.addPoint(s.x + a + s.r, s.y, 0)
    tip_left = occ.addPoint(s.x - a - s.r, s.y, 0)

    # A 180 deg arc given by start/centre/end is ambiguous (which side?), so
    # pass a point ON the arc instead (center=False).
    bottom = occ.addLine(p1, p2)
    right = occ.addCircleArc(p2, tip_right, p3, center=False)
    top = occ.addLine(p3, p4)
    left = occ.addCircleArc(p4, tip_left, p1, center=False)

    # the tips were only construction helpers; otherwise they'd become stray mesh nodes
    occ.remove([(0, tip_right), (0, tip_left)])
    loop = occ.addCurveLoop([bottom, right, top, left])
    surf = occ.addPlaneSurface([loop])

    if s.angle:
        occ.rotate([(2, surf)], s.x, s.y, 0, 0, 0, 1, s.angle)
    return surf


def gen_meshes(
    box: tuple[float, float, float, float],  # x_min, y_min, x_max, y_max
    shapes: list[Stadium],  # Can add other shapes later
    h_far: float,  # element size far from the objects
    h_shape: float,  # element size on/near the object boundaries
    out_dir: str | Path = "mesh/generated",
) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # set to 1 to see gmsh's log
    gmsh.model.add("room")
    occ = gmsh.model.occ

    # --- 1. geometry --------------------------------------------------------
    x0, y0, x1, y1 = box
    air = occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0)

    objs = []
    for s in shapes:
        objs.append(add_stadium(s))
    obj_dimtags = [(2, t) for t in objs]

    # fragment = split everything at intersections so that the air and the
    # objects share their interface curves/nodes (a "conforming" mesh).
    # Surfaces get new tags; out_map[0] lists the pieces of the box and
    # out_map[i + 1] the pieces of shapes[i]. The box overlaps every shape, so
    # its pieces include the shape interiors too -> air = box pieces owned by no shape.
    _, out_map = occ.fragment([(2, air)], obj_dimtags)

    owner: dict[int, Stadium] = {}  # surface tag -> shape it came from
    for s, pieces in zip(shapes, out_map[1:]):
        for _, t in pieces:
            if t in owner:
                raise ValueError(
                    f"shapes overlap, domain would be ambiguous:\n  {owner[t]}\n  {s}"
                )
            owner[t] = s
    obj_surfs = list(owner)
    air_surfs = [t for _, t in out_map[0] if t not in owner]

    occ.synchronize()  # push the CAD kernel's geometry into the gmsh model

    # Object boundary curves (used for local refinement below)
    obj_curves = [
        c
        for _, c in gmsh.model.getBoundary([(2, t) for t in obj_surfs], oriented=False)
    ]

    # domain -> surfaces carrying that label
    groups: dict[Domain, list[int]] = {Domain.AIR: air_surfs}
    for t, s in owner.items():
        groups.setdefault(s.domain, []).append(t)

    # --- 2. physical groups = domain labels you get back later ---------------
    for dom, surfs in groups.items():
        gmsh.model.addPhysicalGroup(2, surfs, tag=int(dom), name=dom.name.lower())

    # --- 3. mesh size control ------------------------------------------------
    # Distance field: distance from every point to the object curves.
    # Threshold field: size h_shape within dist_min, ramping to h_far at dist_max.
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", obj_curves)
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)
    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", h_shape)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 2 * h_shape)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax", 2 * h_shape + 5 * h_far)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)
    # let the field alone decide the size (ignore point sizes / curvature)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # --- 4. mesh -------------------------------------------------------------
    gmsh.model.mesh.generate(2)

    # --- 5. extract arrays ----------------------------------------------------
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    xy = coords.reshape(-1, 3)[:, :2]
    # gmsh tags are 1-based and not guaranteed contiguous -> map tag -> 0-based row
    tag_to_idx = np.full(node_tags.max() + 1, -1, dtype=int)
    tag_to_idx[node_tags] = np.arange(len(node_tags))

    tris, doms = [], []
    for dom, surfs in groups.items():
        for surf in surfs:
            # element type 2 = 3-node triangle
            _, elem_nodes = gmsh.model.mesh.getElementsByType(2, surf)
            t = tag_to_idx[elem_nodes.astype(int)].reshape(-1, 3)
            tris.append(t)
            doms.append(np.full(len(t), int(dom)))
    tris = np.vstack(tris)
    doms = np.concatenate(doms)

    gmsh.finalize()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "nodes.txt", xy)
    np.savetxt(out_dir / "elements.txt", tris, fmt="%d")
    np.savetxt(out_dir / "domains.txt", doms, fmt="%d")
    print(f"{len(xy)} nodes, {len(tris)} triangles -> {out_dir}")
