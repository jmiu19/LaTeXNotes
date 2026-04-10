import numpy as np
import matplotlib.pyplot as plt

# Radial profile (same as before):
# cos(theta) = (r^2 - λ^2)/(r^2 + λ^2),  sin(theta) = 2λr/(r^2 + λ^2)
def theta_profile(r, lam=1.0):
    r2 = r * r
    denom = r2 + lam * lam
    cos_th = (r2 - lam * lam) / denom
    sin_th = (2.0 * lam * r) / denom
    return sin_th, cos_th

# Texture:
# n = (sinθ cosφ, sinθ sinφ, cosθ) with φ = W*α + γ
def n_texture(x, y, lam=1.0, W=1, gamma=0.0):
    r = np.sqrt(x*x + y*y)
    alpha = np.arctan2(y, x)
    sin_th, cos_th = theta_profile(r, lam=lam)
    phi = W * alpha + gamma

    nx = sin_th * np.cos(phi)
    ny = sin_th * np.sin(phi)
    nz = cos_th
    return nx, ny, nz

def vortex_number_W(lam=1.0, W_in=1, gamma=0.0, nalpha=40000):
    # W = (1/2π) ∫ dα ∂α ϕ along a large circle
    R = 5.0 * lam
    alpha = np.linspace(0, 2*np.pi, nalpha, endpoint=False)
    x = R * np.cos(alpha)
    y = R * np.sin(alpha)
    nx, ny, _ = n_texture(x, y, lam=lam, W=W_in, gamma=gamma)
    phi = np.unwrap(np.arctan2(ny, nx))
    return (phi[-1] - phi[0]) / (2*np.pi)

def skyrmion_number_N(lam=1.0, W=1, gamma=0.0, L=10.0, Ngrid=601):
    """
    N = (1/4π) ∫ n · (∂x n × ∂y n) dx dy on [-L,L]^2.
    """
    x = np.linspace(-L, L, Ngrid)
    y = np.linspace(-L, L, Ngrid)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="xy")

    nx, ny, nz = n_texture(X, Y, lam=lam, W=W, gamma=gamma)
    n = np.stack([nx, ny, nz], axis=-1)

    dn_dx = (n[:, 2:, :] - n[:, :-2, :]) / (2*dx)
    dn_dy = (n[2:, :, :] - n[:-2, :, :]) / (2*dy)

    n_int = n[1:-1, 1:-1, :]
    dn_dx_int = dn_dx[1:-1, :, :]
    dn_dy_int = dn_dy[:, 1:-1, :]

    density = np.einsum("...i,...i->...", n_int, np.cross(dn_dx_int, dn_dy_int))
    N = (1.0 / (4*np.pi)) * np.sum(density) * dx * dy
    return N

def plot_texture(lam=1.0, W=1, gamma=0.0, L=6.0, Ngrid=241, quiver_stride=10):
    x = np.linspace(-L, L, Ngrid)
    y = np.linspace(-L, L, Ngrid)
    X, Y = np.meshgrid(x, y, indexing="xy")
    nx, ny, nz = n_texture(X, Y, lam=lam, W=W, gamma=gamma)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    im = axes[0].imshow(nz, extent=[-L, L, -L, L], origin="lower", aspect="equal")
    axes[0].set_title(rf"$n_z(x,y)$ (W={W})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    s = quiver_stride
    axes[0].quiver(X[::s, ::s], Y[::s, ::s], nx[::s, ::s], ny[::s, ::s],
                   pivot="mid", scale=25)

    phi = np.arctan2(ny, nx)
    im2 = axes[1].imshow(phi, extent=[-L, L, -L, L], origin="lower", aspect="equal")
    axes[1].set_title(r"In-plane angle $\varphi(x,y)$")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    return fig, axes

if __name__ == "__main__":
    lam = 1.0

    # --- Choose W=+2 with P=-1 profile -> N = -2 ---
    W = 2
    gamma = 0.0

    fig, axes = plot_texture(lam=lam, W=W, gamma=gamma, L=6.0, Ngrid=241, quiver_stride=10)
    plt.show()

    W_meas = vortex_number_W(lam=lam, W_in=W, gamma=gamma)
    N_meas = skyrmion_number_N(lam=lam, W=W, gamma=gamma, L=10.0, Ngrid=601)

    print(f"Measured vortex number W ≈ {W_meas:.6f} (target {W})")
    print(f"Measured skyrmion number N ≈ {N_meas:.6f} (target -2)")
    print(f"Expected (P=-1) => N = -W = {-W}")
