"""
2D Random Walk & Brownian Diffusion Regimes
Simulates ensembles of N = 1000 2D random walkers under three regimes:
  1. Free diffusion: linear mean-squared displacement <r^2(t)> ~ t
  2. Confined diffusion: reflective box boundary conditions
  3. Harmonic trap potential: V(x, y) = k (x^2 + y^2)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def potential_energy(x: np.ndarray, y: np.ndarray, k: float) -> np.ndarray:
    """Compute conservative harmonic potential V(x, y) = k * (x^2 + y^2)."""
    return k * (x**2 + y**2)


def apply_boundary_conditions(
    x: np.ndarray, y: np.ndarray, box_size: float, mode: str = "reflecting"
) -> tuple[np.ndarray, np.ndarray]:
    """Apply reflecting or periodic boundary conditions on [-box_size, box_size]^2."""
    if mode == "reflecting":
        return np.clip(x, -box_size, box_size), np.clip(y, -box_size, box_size)
    if mode == "periodic":
        x_p = (x + box_size) % (2 * box_size) - box_size
        y_p = (y + box_size) % (2 * box_size) - box_size
        return x_p, y_p
    return x, y


def simulate_random_walk_2d(
    n_walkers: int = 1000,
    steps: int = 500,
    step_size: float = 1.0,
    k_trap: float = 0.01,
    box_size: float = 20.0,
    track_energy: bool = False,
    lattice: bool = False,
    boundary: str = "none",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate 2D random walk trajectories and ensemble potential energy."""
    x = np.zeros((n_walkers, steps))
    y = np.zeros((n_walkers, steps))
    mean_energies = np.zeros(steps)

    for t in range(1, steps):
        if lattice:
            directions = np.random.choice([0, 1, 2, 3], n_walkers)
            dx = np.choose(directions, [step_size, -step_size, 0.0, 0.0])
            dy = np.choose(directions, [0.0, 0.0, step_size, -step_size])
        else:
            angles = np.random.uniform(0.0, 2.0 * np.pi, n_walkers)
            dx = step_size * np.cos(angles)
            dy = step_size * np.sin(angles)

        x[:, t] = x[:, t - 1] + dx
        y[:, t] = y[:, t - 1] + dy

        if boundary in ("reflecting", "periodic"):
            x[:, t], y[:, t] = apply_boundary_conditions(
                x[:, t], y[:, t], box_size, boundary
            )

        if track_energy:
            mean_energies[t] = np.mean(potential_energy(x[:, t], y[:, t], k_trap))

    return x, y, mean_energies


def mean_squared_displacement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute ensemble-averaged mean-squared displacement <r^2(t)>."""
    return np.mean(x**2 + y**2, axis=0)


def main():
    n_walkers = 1000
    steps = 500
    step_size = 1.0
    box_size = 20.0
    k_trap = 0.01

    x_free, y_free, _ = simulate_random_walk_2d(
        n_walkers, steps, step_size, boundary="none"
    )
    x_conf, y_conf, _ = simulate_random_walk_2d(
        n_walkers, steps, step_size, box_size=box_size, boundary="reflecting"
    )
    x_harm, y_harm, energies = simulate_random_walk_2d(
        n_walkers, steps, step_size, k_trap=k_trap, track_energy=True
    )

    msd_free = mean_squared_displacement(x_free, y_free)
    msd_conf = mean_squared_displacement(x_conf, y_conf)
    msd_harm = mean_squared_displacement(x_harm, y_harm)

    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for idx in range(5):
        axes[0, 0].plot(x_free[idx], y_free[idx], alpha=0.85)
        axes[0, 1].plot(x_conf[idx], y_conf[idx], alpha=0.85)
        axes[0, 2].plot(x_harm[idx], y_harm[idx], alpha=0.85)

    axes[0, 0].set_title("Sample Trajectories — Free Diffusion")
    axes[0, 1].set_title(f"Sample Trajectories — Confined ($L = \\pm{box_size}$)")
    axes[0, 2].set_title("Sample Trajectories — Harmonic Potential")
    for ax in axes[0]:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[1, 0].plot(msd_free, color="tab:blue", label="Simulated $\\langle r^2(t) \\rangle$")
    axes[1, 0].plot(
        np.arange(steps) * step_size**2,
        "k--",
        alpha=0.7,
        label="Theory $N_{\\mathrm{steps}} a^2$",
    )
    axes[1, 0].set_title("Mean-Squared Displacement — Free")

    axes[1, 1].plot(msd_conf, color="tab:orange", label="Confined $\\langle r^2(t) \\rangle$")
    axes[1, 1].set_title("Mean-Squared Displacement — Confined")

    axes[1, 2].plot(energies, color="tab:green", label="Mean Potential $\\langle V(t) \\rangle$")
    axes[1, 2].set_title("Mean Harmonic Potential Energy")

    for ax in axes[1]:
        ax.set_xlabel("Step $t$")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/random_walk_2d.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
