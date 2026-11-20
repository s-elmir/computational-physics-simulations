"""
2D Ferromagnetic Ising Model via Metropolis Monte Carlo
Computes magnetization per site <|m|>, magnetic susceptibility chi,
mean energy per spin <e>, and specific heat Cv across the Onsager
phase transition (Tc ≈ 2.269 J/kB) with periodic boundary conditions.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt


def init_lattice(L: int) -> np.ndarray:
    """Initialize an L x L lattice with random spins in {-1, +1}."""
    return np.random.choice([-1, 1], p=[0.5, 0.5], size=(L, L))


def delta_energy(i: int, j: int, lattice: np.ndarray, L: int, J: float) -> float:
    """Compute energy variation dE for flipping spin (i, j) under PBCs."""
    neighbors_sum = (
        lattice[(i - 1) % L, j]
        + lattice[(i + 1) % L, j]
        + lattice[i, (j - 1) % L]
        + lattice[i, (j + 1) % L]
    )
    return 2.0 * J * lattice[i, j] * neighbors_sum


def mc_move(lattice: np.ndarray, L: int, J: float, proba_dE: dict) -> None:
    """Perform a single Metropolis spin-flip trial in place."""
    i, j = np.random.randint(L), np.random.randint(L)
    dE = delta_energy(i, j, lattice, L, J)
    if dE <= 0 or np.random.random() < proba_dE[dE]:
        lattice[i, j] = -lattice[i, j]


def total_energy(lattice: np.ndarray, J: float) -> float:
    """Compute total Ising Hamiltonian energy H = -J * sum_<ij> S_i S_j."""
    right_neighbors = np.roll(lattice, -1, axis=1)
    down_neighbors = np.roll(lattice, -1, axis=0)
    return -J * np.sum(lattice * (right_neighbors + down_neighbors))


def simulate_ising_at_T(
    T: float, L: int, J: float, K: int, n_therm: int, n_meas: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run Metropolis simulation at temperature T and return |M| and E series."""
    n_sites = L * L
    beta = 1.0 / T
    proba_dE = {dE: np.exp(-beta * dE) for dE in 2 * J * np.arange(-4, 6, 2)}

    lattice = init_lattice(L)
    mag_series = np.zeros(n_meas)
    ene_series = np.zeros(n_meas)

    # Thermalization phase
    for _ in range(n_therm):
        for _ in range(n_sites):
            mc_move(lattice, L, J, proba_dE)

    # Measurement phase
    for step in range(n_meas):
        for _ in range(n_sites * K):
            mc_move(lattice, L, J, proba_dE)
        mag_series[step] = abs(np.sum(lattice))
        ene_series[step] = total_energy(lattice, J)

    return mag_series, ene_series


def main():
    L = 20               # Lattice linear dimension (L x L)
    n_sites = L * L
    J = 1.0              # Ferromagnetic coupling constant
    K = 1                # Sweeps between measurements
    n_therm = 1000       # Thermalization sweeps
    n_meas = 3000        # Measurement samples per temperature
    Tc_onsager = 2.0 * J / np.log(1.0 + np.sqrt(2.0))  # Exact 2D Tc ≈ 2.269

    temperatures = np.linspace(0.5, 4.0, 22)
    m_mean = np.zeros_like(temperatures)
    chi = np.zeros_like(temperatures)
    e_mean = np.zeros_like(temperatures)
    cv = np.zeros_like(temperatures)

    t_start = time.perf_counter()
    for idx, T in enumerate(temperatures):
        mag, ene = simulate_ising_at_T(T, L, J, K, n_therm, n_meas)
        m = mag / n_sites
        e = ene / n_sites

        m_mean[idx] = np.mean(m)
        chi[idx] = n_sites * (np.mean(m**2) - m_mean[idx] ** 2) / T
        e_mean[idx] = np.mean(e)
        cv[idx] = n_sites * (np.mean(e**2) - e_mean[idx] ** 2) / (T**2)

        print(
            f"T = {T:4.2f} | <|m|> = {m_mean[idx]:.3f} | "
            f"chi = {chi[idx]:6.2f} | <e> = {e_mean[idx]:6.3f}"
        )

    print(f"Total runtime: {(time.perf_counter() - t_start) / 60:.2f} min")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    plots = [
        (axes[0, 0], m_mean, "o-", "tab:blue", "Magnetization per site $\\langle |m| \\rangle$"),
        (axes[0, 1], chi, "s-", "tab:orange", "Magnetic Susceptibility $\\chi$"),
        (axes[1, 0], e_mean, "^-", "tab:red", "Mean Energy per spin $\\langle e \\rangle / J$"),
        (axes[1, 1], cv, "d-", "tab:green", "Specific Heat $C_v / k_B$"),
    ]

    for ax, data, style, color, ylabel in plots:
        ax.plot(temperatures, data, style, color=color, markersize=5)
        ax.axvline(
            Tc_onsager, color="k", linestyle="--", alpha=0.7, label=f"$T_c \\approx {Tc_onsager:.3f}$"
        )
        ax.set_xlabel("Temperature $T$ ($J / k_B$)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"2D Ising Model Phase Transition ($N = {L}\\times{L}$)", fontsize=14)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/ising_2d_observables.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
