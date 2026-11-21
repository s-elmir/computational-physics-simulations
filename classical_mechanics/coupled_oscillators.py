"""
1D Coupled Harmonic Oscillator Chain (Velocity-Verlet Integration)
Simulates longitudinal oscillations of N coupled masses connected by Hookean
springs with free boundary conditions, verifying energy conservation over time.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def compute_forces(positions: np.ndarray, eq_positions: np.ndarray, k: float) -> np.ndarray:
    """Compute restoring spring forces on each mass from displacements u_i = x_i - x_i^0."""
    u = positions - eq_positions
    forces = np.zeros_like(u)
    forces[1:-1] = -k * (u[1:-1] - u[:-2]) + k * (u[2:] - u[1:-1])
    forces[0] = k * (u[1] - u[0])
    forces[-1] = -k * (u[-1] - u[-2])
    return forces


def kinetic_energy(velocities: np.ndarray, mass: float) -> float:
    """Compute total kinetic energy E_k = 0.5 * m * sum(v_i^2)."""
    return 0.5 * mass * float(np.sum(velocities**2))


def potential_energy(positions: np.ndarray, eq_positions: np.ndarray, k: float) -> float:
    """Compute total elastic potential energy E_p = 0.5 * k * sum((u_{i+1} - u_i)^2)."""
    u = positions - eq_positions
    du = np.diff(u)
    return 0.5 * k * float(np.sum(du**2))


def main():
    dt = 0.01
    mass = 1.0
    n_particles = 5
    k = 4.0
    n_steps = 1500
    time = np.arange(n_steps) * dt

    eq_positions = np.arange(n_particles, dtype=float)
    positions = np.zeros((n_particles, n_steps))
    velocities = np.zeros((n_particles, n_steps))
    e_kin = np.zeros(n_steps)
    e_pot = np.zeros(n_steps)

    # Initial perturbation on the first particle
    positions[:, 0] = eq_positions.copy()
    positions[0, 0] -= 0.4
    velocities[:, 0] = 0.0

    forces = compute_forces(positions[:, 0], eq_positions, k)
    e_kin[0] = kinetic_energy(velocities[:, 0], mass)
    e_pot[0] = potential_energy(positions[:, 0], eq_positions, k)

    # Velocity-Verlet integration loop
    for t in range(1, n_steps):
        positions[:, t] = (
            positions[:, t - 1] + velocities[:, t - 1] * dt + 0.5 * (forces / mass) * dt**2
        )
        new_forces = compute_forces(positions[:, t], eq_positions, k)
        velocities[:, t] = velocities[:, t - 1] + 0.5 * ((forces + new_forces) / mass) * dt
        forces = new_forces

        e_kin[t] = kinetic_energy(velocities[:, t], mass)
        e_pot[t] = potential_energy(positions[:, t], eq_positions, k)

    e_tot = e_kin + e_pot

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for i in range(n_particles):
        ax1.plot(time, positions[i], label=f"Mass {i + 1}")
    ax1.set_title(f"Coupled Harmonic Chain ($N = {n_particles}$, Velocity-Verlet)")
    ax1.set_xlabel("Time $t$ (s)")
    ax1.set_ylabel("Position $x_i(t)$")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(time, e_kin, label="Kinetic Energy $E_k$")
    ax2.plot(time, e_pot, label="Potential Energy $E_p$")
    ax2.plot(time, e_tot, "k--", linewidth=1.8, label="Total Energy $E_m$")
    ax2.set_title("Symplectic Energy Conservation")
    ax2.set_xlabel("Time $t$ (s)")
    ax2.set_ylabel("Energy")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/coupled_oscillators.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
