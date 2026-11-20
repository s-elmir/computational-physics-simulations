"""
2D Binary Alloy Quenching via Kawasaki Spin-Exchange Monte Carlo
Simulates phase separation in a conserved-order-parameter 2D binary alloy (A/B)
during thermal quenching from high to low temperature:
  - p = 0.50: Spinodal decomposition (interconnected labyrinthine domains)
  - p = 0.10: Nucleation and growth (minority droplets in a majority matrix)
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt


def init_lattice(L: int, p_ratio: float = 0.5) -> np.ndarray:
    """Initialize an L x L lattice with concentration p_ratio of +1 (A) and 1 - p_ratio of -1 (B)."""
    return np.random.choice([1, -1], p=[p_ratio, 1.0 - p_ratio], size=(L, L))


def delta_energy_exchange(
    i1: int, j1: int, i2: int, j2: int, lattice: np.ndarray, L: int, J: float
) -> float:
    """Compute energy difference dE for swapping two nearest-neighbor unlike spins under PBCs."""
    sum_neighbors_1 = (
        lattice[(i1 - 1) % L, j1]
        + lattice[(i1 + 1) % L, j1]
        + lattice[i1, (j1 - 1) % L]
        + lattice[i1, (j1 + 1) % L]
    )
    sum_neighbors_2 = (
        lattice[(i2 - 1) % L, j2]
        + lattice[(i2 + 1) % L, j2]
        + lattice[i2, (j2 - 1) % L]
        + lattice[i2, (j2 + 1) % L]
    )
    return 2.0 * J * lattice[i1, j1] * (sum_neighbors_1 - sum_neighbors_2) + 4.0 * J


def get_proba_dict(beta: float, J: float) -> dict:
    """Pre-compute Boltzmann acceptance probabilities for possible dE values."""
    possible_dEs = np.arange(-16, 20, 4) * J
    return {float(dE): np.exp(-dE * beta) for dE in possible_dEs}


def kawasaki_move(
    lattice: np.ndarray, L: int, J: float, beta: float, proba_dE: dict
) -> None:
    """Attempt a single nearest-neighbor spin-exchange (Kawasaki dynamics) in place."""
    i1, j1 = np.random.randint(L), np.random.randint(L)
    neighbors = [
        ((i1 - 1) % L, j1),
        ((i1 + 1) % L, j1),
        (i1, (j1 - 1) % L),
        (i1, (j1 + 1) % L),
    ]
    i2, j2 = neighbors[np.random.randint(4)]

    if lattice[i1, j1] != lattice[i2, j2]:
        dE = float(delta_energy_exchange(i1, j1, i2, j2, lattice, L, J))
        if dE <= 0.0:
            lattice[i1, j1], lattice[i2, j2] = -lattice[i1, j1], -lattice[i2, j2]
        else:
            proba = proba_dE.get(dE)
            if proba is None:
                proba = np.exp(-beta * dE)
            if np.random.random() < proba:
                lattice[i1, j1], lattice[i2, j2] = -lattice[i1, j1], -lattice[i2, j2]


def simulate_quench(
    p_ratio: float,
    T_init: float = 5.0,
    T_final: float = 0.1,
    T_steps: int = 100,
    L: int = 50,
    J: float = 1.0,
    sweeps_per_T: int = 1,
    regime_label: str = "",
) -> None:
    """Run a linear temperature quench from T_init to T_final and plot snapshots."""
    t_start = time.perf_counter()
    lattice = init_lattice(L, p_ratio)
    temperatures = np.linspace(T_init, T_final, T_steps)
    n_sites = L * L

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    snapshot_indices = {0: 0, T_steps // 2: 1, T_steps - 1: 2}

    for idx, T in enumerate(temperatures):
        beta = 1.0 / T
        proba_dE = get_proba_dict(beta, J)

        for _ in range(n_sites * sweeps_per_T):
            kawasaki_move(lattice, L, J, beta, proba_dE)

        if idx in snapshot_indices:
            ax = axes[snapshot_indices[idx]]
            ax.imshow(lattice, cmap="bwr", vmin=-1, vmax=1)
            ax.set_title(f"$T = {T:.2f}\\,J/k_B$")
            ax.axis("off")

    title_suffix = f" — {regime_label}" if regime_label else ""
    fig.suptitle(
        f"2D Binary Alloy Quench ($p = {p_ratio:.2f}$, $L = {L}$){title_suffix}",
        fontsize=13,
    )
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/binary_alloy_p{int(p_ratio * 100)}.png", dpi=200)
    print(f"Quench p={p_ratio:.2f} completed in {time.perf_counter() - t_start:.2f} s")
    plt.show()


def main():
    # Case 1: 50/50 mixture -> Spinodal decomposition
    simulate_quench(
        p_ratio=0.50,
        T_init=5.0,
        T_final=0.1,
        T_steps=100,
        L=50,
        regime_label="Spinodal Decomposition",
    )

    # Case 2: 10/90 mixture -> Nucleation and growth
    simulate_quench(
        p_ratio=0.10,
        T_init=5.0,
        T_final=0.1,
        T_steps=100,
        L=50,
        regime_label="Nucleation & Growth",
    )


if __name__ == "__main__":
    main()
