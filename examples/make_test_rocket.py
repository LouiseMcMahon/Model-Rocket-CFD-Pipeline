"""Generate a simple watertight model rocket STL for testing the pipeline.

Tangent-ogive nose, straight body tube, trapezoidal fins. Written in millimetres with the
nose pointing +z, like a typical CAD export, so the pipeline's unit and axis detection get
exercised.

    python examples/make_test_rocket.py models/test_rocket.stl --fins 4
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh


def ogive_profile(radius: float, length: float, n: int = 40) -> np.ndarray:
    rho = (radius**2 + length**2) / (2 * radius)
    z = np.linspace(0, length, n)
    r = np.sqrt(rho**2 - (length - z) ** 2) + radius - rho
    r[0] = 0.0
    return np.column_stack([r, z])


def make_rocket(diameter=41.6, nose_len=125.0, body_len=480.0, n_fins=4,
                root=90.0, tip=40.0, span=50.0, sweep=40.0, thickness=3.0) -> trimesh.Trimesh:
    R = diameter / 2
    L = nose_len + body_len
    # Profile from tail centre, up the body, over the nose to the tip (z = 0 is the tail).
    nose = ogive_profile(R, nose_len)
    nose_z = L - nose[:, 1]                        # tip at z = L
    profile = np.vstack([[0, 0], [R, 0], np.column_stack([nose[::-1, 0], nose_z[::-1]])])
    body = trimesh.creation.revolve(profile, sections=96)

    # Fin in the x-z plane: root along the body from the tail, swept leading edge.
    x_in = R - 1.0
    pts2d = np.array([[x_in, 0], [x_in, root], [R + span, sweep + tip], [R + span, sweep]])
    verts = np.vstack([np.column_stack([pts2d[:, 0], np.full(4, y), pts2d[:, 1]])
                       for y in (-thickness / 2, thickness / 2)])
    fin = trimesh.Trimesh(verts).convex_hull
    parts = [body]
    for k in range(n_fins):
        f = fin.copy()
        f.apply_transform(trimesh.transformations.rotation_matrix(2 * np.pi * k / n_fins, [0, 0, 1]))
        parts.append(f)
    return trimesh.boolean.union(parts, engine="manifold")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fins", type=int, default=4)
    args = ap.parse_args()
    rocket = make_rocket(n_fins=args.fins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rocket.export(args.out)
    print(f"Wrote {args.out}: {len(rocket.faces)} triangles, watertight={rocket.is_watertight}, "
          f"extents (mm)={np.round(rocket.extents, 1)}")
