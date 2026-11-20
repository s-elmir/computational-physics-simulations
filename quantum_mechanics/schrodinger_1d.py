"""
1D Time-Dependent Schrödinger Equation Solver
Compares the numerical stability and norm conservation (unitarity) of
Explicit Euler vs. Crank-Nicolson finite-difference schemes for a Gaussian
wavepacket scattering off a potential step.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# --- Physical & Numerical Parameters ---
L = 100.0          # Spatial domain length
Nx = 500           # Number of spatial grid points
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)

dt = 0.01          # Time step
Nt = 500           # Number of time iterations
hbar = 1.0         # Dimensionless Planck constant
mass = 1.0         # Particle mass

# --- Potential V(x): Potential Step / Barrier ---
V = np.zeros(Nx)
V[Nx // 2 : Nx // 2 + 20] = 0.5

# --- Initial Condition: Normalized Gaussian Wavepacket ---
x0 = 30.0
sigma = 5.0
k0 = 1.5           # Initial momentum
psi0 = np.exp(-0.5 * ((x - x0) / sigma) ** 2) * np.exp(1j * k0 * x)

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
psi0 /= np.sqrt(_trapz(np.abs(psi0) ** 2, x))


def solve_step(psi: np.ndarray, method: str, v_pot: np.ndarray) -> np.ndarray:
    """Advance the wavefunction by one time step dt."""
    if method == "euler_explicit":
        d2psi = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / dx**2
        h_psi = -0.5 * (hbar**2 / mass) * d2psi + v_pot * psi
        return psi - (1j * dt / hbar) * h_psi

    if method == "crank_nicolson":
        alpha = (1j * hbar * dt) / (4 * mass * dx**2)
        diag_main = 1 + (1j * dt / (2 * hbar)) * (hbar**2 / (mass * dx**2) + v_pot)

        rhs = (
            1 - (1j * dt / (2 * hbar)) * (hbar**2 / (mass * dx**2) + v_pot)
        ) * psi + alpha * (np.roll(psi, 1) + np.roll(psi, -1))

        # Tridiagonal banded matrix for scipy.linalg.solve_banded
        ab = np.zeros((3, Nx), dtype=complex)
        ab[0, 1:] = -alpha      # Upper diagonal
        ab[1, :] = diag_main    # Main diagonal
        ab[2, :-1] = -alpha     # Lower diagonal

        return solve_banded((1, 1), ab, rhs)

    raise ValueError(f"Unknown integration method: {method}")


def main():
    methods = ["euler_explicit", "crank_nicolson"]
    final_densities = {}
    norm_histories = {}

    for method in methods:
        psi = psi0.copy()
        norms = []
        for step in range(Nt):
            psi = solve_step(psi, method, V)
            current_norm = _trapz(np.abs(psi) ** 2, x)
            norms.append(current_norm)

            # Stop early if explicit scheme diverges
            if current_norm > 100:
                print(f"[{method}] Diverged at t = {step * dt:.2f}")
                norms.extend([np.nan] * (Nt - len(norms)))
                break

        final_densities[method] = np.abs(psi) ** 2
        norm_histories[method] = norms

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(x, np.abs(psi0) ** 2, "k:", label="Initial $|\\psi(x, 0)|^2$")
    ax1.plot(
        x,
        final_densities["crank_nicolson"],
        label="Crank-Nicolson $|\\psi(x, T)|^2$",
    )
    ax1.fill_between(
        x, 0, V * 0.1, color="gray", alpha=0.3, label="Potential $V(x)$ (scaled)"
    )
    ax1.set_title("1D Gaussian Wavepacket Scattering (Crank-Nicolson)")
    ax1.set_xlabel("Position $x$")
    ax1.set_ylabel("Probability Density")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(norm_histories["euler_explicit"], label="Explicit Euler (Unstable)")
    ax2.plot(norm_histories["crank_nicolson"], label="Crank-Nicolson (Unitary)")
    ax2.set_yscale("log")
    ax2.set_title("Numerical Stability: Wavefunction Norm Conservation")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Total Norm $\\int |\\psi|^2 dx$ (log scale)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/schrodinger_1d.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
