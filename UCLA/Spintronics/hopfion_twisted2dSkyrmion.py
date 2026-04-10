"""
Hopfion visualization as a twisted skyrmion tube (torus-like domain wall surface).

Key idea:
- Build m(x,y,z) using the standard Hopf map R^3 -> S^2 (Q=1).
- Plot the isosurface m_z = 0 (often a torus-like surface).
- Color the surface by phi = atan2(m_y, m_x) to show the twist along the ring.

Dependencies:
  pip install numpy matplotlib scikit-image
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from skimage.measure import marching_cubes
except ImportError as e:
    raise SystemExit(
        "Missing dependency: scikit-image\n"
        "Install with: pip install scikit-image"
    ) from e


def hopfion_field(N=96, L=3.0, scale=1.0, dtype=np.float32):
    """
    Analytic Q=1 Hopf map magnetization field m: R^3 -> S^2.

    Parameters
    ----------
    N : int
        Grid points per axis.
    L : float
        Half-size of the cubic domain: x,y,z in [-L, L].
    scale : float
        Spatial scale factor. Larger scale -> Hopfion features appear "smaller" in the box.
    dtype : numpy dtype
        Float dtype.

    Returns
    -------
    x, y, z : 1D arrays
        Coordinate axes.
    m : array, shape (N,N,N,3)
        Unit magnetization field.
    """
    x = np.linspace(-L, L, N, dtype=dtype)
    y = np.linspace(-L, L, N, dtype=dtype)
    z = np.linspace(-L, L, N, dtype=dtype)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Rescale coordinates (controls effective size)
    Xs, Ys, Zs = X / scale, Y / scale, Z / scale
    r2 = Xs * Xs + Ys * Ys + Zs * Zs
    denom = r2 + 1.0

    # Stereographic inverse: R^3 -> S^3 embedded in R^4
    X1 = 2.0 * Xs / denom
    X2 = 2.0 * Ys / denom
    X3 = 2.0 * Zs / denom
    X4 = (r2 - 1.0) / denom

    # Complex coordinates on S^3: u, v with |u|^2 + |v|^2 = 1
    u = X1.astype(np.complex64) + 1j * X2.astype(np.complex64)
    v = X3.astype(np.complex64) + 1j * X4.astype(np.complex64)

    # Hopf map to S^2
    uv = u * np.conj(v)
    mx = (2.0 * uv.real).astype(dtype)
    my = (2.0 * uv.imag).astype(dtype)
    mz = (np.abs(u) ** 2 - np.abs(v) ** 2).astype(dtype)

    m = np.stack([mx, my, mz], axis=-1)

    # Numerical renormalization (should already be ~1)
    norm = np.sqrt(np.sum(m * m, axis=-1, keepdims=True))
    m = m / norm
    return x, y, z, m


def trilinear_interp(vol, pts):
    """
    Trilinear interpolation of a 3D scalar volume at fractional index points.

    vol: (Nx,Ny,Nz)
    pts: (M,3) fractional indices in [0..N-1]
    """
    Nx, Ny, Nz = vol.shape
    pts = np.asarray(pts, dtype=np.float64)

    xi, yi, zi = pts[:, 0], pts[:, 1], pts[:, 2]
    x0 = np.floor(xi).astype(int)
    y0 = np.floor(yi).astype(int)
    z0 = np.floor(zi).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Clamp indices
    x0 = np.clip(x0, 0, Nx - 1)
    x1 = np.clip(x1, 0, Nx - 1)
    y0 = np.clip(y0, 0, Ny - 1)
    y1 = np.clip(y1, 0, Ny - 1)
    z0 = np.clip(z0, 0, Nz - 1)
    z1 = np.clip(z1, 0, Nz - 1)

    xd = xi - x0
    yd = yi - y0
    zd = zi - z0

    c000 = vol[x0, y0, z0]
    c100 = vol[x1, y0, z0]
    c010 = vol[x0, y1, z0]
    c110 = vol[x1, y1, z0]
    c001 = vol[x0, y0, z1]
    c101 = vol[x1, y0, z1]
    c011 = vol[x0, y1, z1]
    c111 = vol[x1, y1, z1]

    c00 = c000 * (1 - xd) + c100 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd
    return c


def main():
    # --- Tunables ---
    N = 96          # increase for smoother surfaces (costs memory/time)
    L = 3.0         # half-box size
    scale = 1.0     # adjust to expand/shrink Hopfion features in the box
    iso_level = 0.0 # m_z = 0 surface
    step_size = 2   # marching cubes subsampling (1 = best quality, slower)

    x, y, z, m = hopfion_field(N=N, L=L, scale=scale)
    mx, my, mz = m[..., 0], m[..., 1], m[..., 2]
    phi = np.arctan2(my, mx).astype(np.float32)  # twist angle

    dx = float(x[1] - x[0])
    origin = np.array([-L, -L, -L], dtype=np.float64)

    # Extract isosurface of mz
    verts, faces, normals, values = marching_cubes(
        mz, level=iso_level, spacing=(dx, dx, dx), step_size=step_size
    )
    verts = verts + origin  # shift from [0..] coords to [-L..L]

    # Color surface by phi at vertices (interpolate phi on the grid)
    verts_idx = (verts - origin) / dx  # fractional indices in array coordinates
    phi_v = trilinear_interp(phi, verts_idx)

    # Map phi -> colormap (cyclic colormap is nice for angles)
    cmap = cm.twilight
    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    rgba_v = cmap(norm(phi_v))  # per-vertex colors

    # Convert to per-face colors by averaging vertex colors
    face_colors = rgba_v[faces].mean(axis=1)

    # --- Plot ---
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    poly = Poly3DCollection(verts[faces], facecolors=face_colors, linewidths=0.0)
    poly.set_edgecolor("none")
    ax.add_collection3d(poly)

    # Limits + aspect
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Hopfion as a twisted skyrmion tube: isosurface $m_z=0$ colored by $\\phi=\\arctan2(m_y,m_x)$")

    # Colorbar for phi
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label("twist angle ϕ (radians)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
