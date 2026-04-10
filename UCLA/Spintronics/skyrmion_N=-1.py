import numpy as np
import matplotlib.pyplot as plt

# --- Skyrmion field from the thesis ---
# n(x,y) = (2λx, 2λy, r^2 - λ^2) / (r^2 + λ^2)
def skyrmion_n(x, y, lam=1.0):
    r2 = x*x + y*y
    denom = r2 + lam*lam
    nx = 2.0 * lam * x / denom
    ny = 2.0 * lam * y / denom
    nz = (r2 - lam*lam) / denom
    return nx, ny, nz

# --- Helpers for "polarity" P and vortex number W used in Eq. (1.11) ---
def polarity_P(lam=1.0):
    # For this texture: n_z(0) = -1, n_z(∞) = +1  => P = ( -1 - 1 ) / 2 = -1
    nz0 = -1.0
    nzinf = +1.0
    return 0.5 * (nz0 - nzinf)

def vortex_number_W(lam=1.0, nalpha=20000):
    # W = (1/2π) ∫_0^{2π} dα ∂α ϕ, where ϕ is the in-plane angle of (n_x, n_y).
    # For this skyrmion: ϕ = atan2(n_y, n_x) = atan2(y, x) = α  (for r>0), so W=+1.
    # We'll compute it numerically along a circle of radius R.
    R = 5.0 * lam
    alpha = np.linspace(0, 2*np.pi, nalpha, endpoint=False)
    x = R * np.cos(alpha)
    y = R * np.sin(alpha)
    nx, ny, _ = skyrmion_n(x, y, lam=lam)
    phi = np.unwrap(np.arctan2(ny, nx))
    dphi = phi[-1] - phi[0]
    return dphi / (2*np.pi)

# --- Skyrmion number N (topological charge) ---
def skyrmion_number_N(lam=1.0, L=6.0, Ngrid=401):
    """
    Compute N = (1/4π) ∫ n · (∂x n × ∂y n) dx dy on a square [-L,L]^2.
    Uses central differences and the area element dx dy.
    """
    x = np.linspace(-L, L, Ngrid)
    y = np.linspace(-L, L, Ngrid)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="xy")

    nx, ny, nz = skyrmion_n(X, Y, lam=lam)
    n = np.stack([nx, ny, nz], axis=-1)

    # central differences for ∂x n and ∂y n
    dn_dx = (n[:, 2:, :] - n[:, :-2, :]) / (2*dx)
    dn_dy = (n[2:, :, :] - n[:-2, :, :]) / (2*dy)

    # align shapes: interior region
    n_int = n[1:-1, 1:-1, :]
    dn_dx_int = dn_dx[1:-1, :, :]    # remove y-boundary
    dn_dy_int = dn_dy[:, 1:-1, :]    # remove x-boundary

    cross = np.cross(dn_dx_int, dn_dy_int)
    density = np.einsum("...i,...i->...", n_int, cross)  # n · (∂x n × ∂y n)

    N = (1.0 / (4*np.pi)) * np.sum(density) * dx * dy
    return N

# --- Visualization ---
def plot_skyrmion(lam=1.0, L=6.0, Ngrid=201, quiver_stride=8):
    x = np.linspace(-L, L, Ngrid)
    y = np.linspace(-L, L, Ngrid)
    X, Y = np.meshgrid(x, y, indexing="xy")
    nx, ny, nz = skyrmion_n(X, Y, lam=lam)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    # Left: color map of n_z
    im = axes[0].imshow(
        nz, extent=[-L, L, -L, L], origin="lower", aspect="equal"
    )
    axes[0].set_title(r"$n_z(x,y)$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # Overlay in-plane arrows (n_x, n_y)
    s = quiver_stride
    axes[0].quiver(
        X[::s, ::s], Y[::s, ::s],
        nx[::s, ::s], ny[::s, ::s],
        pivot="mid", scale=25
    )

    # Right: in-plane angle phi
    phi = np.arctan2(ny, nx)
    im2 = axes[1].imshow(
        phi, extent=[-L, L, -L, L], origin="lower", aspect="equal"
    )
    axes[1].set_title(r"In-plane angle $\varphi(x,y)=\mathrm{atan2}(n_y,n_x)$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    return fig, axes

if __name__ == "__main__":
    lam = 1.0

    # Visualize
    fig, axes = plot_skyrmion(lam=lam, L=6.0, Ngrid=241, quiver_stride=10)
    plt.show()

    # Compute the quantities in Eq. (1.11): N = P * W
    P = polarity_P(lam=lam)
    W = vortex_number_W(lam=lam)
    N = skyrmion_number_N(lam=lam, L=10.0, Ngrid=601)

    print(f"Polarity P ≈ {P:.6f} (expected -1 for this texture)")
    print(f"Vortex number W ≈ {W:.6f} (expected +1 for this texture)")
    print(f"Skyrmion number N ≈ {N:.6f} (expected -1 for this texture)")
    print(f"P * W ≈ {P*W:.6f}")
