"""
2D Lennard-Jones Fluid Molecular Dynamics Simulation
Integrates equations of motion for N = 64 particles in a reflective box
using the Velocity-Verlet algorithm, with optional uniform gravity and
real-time tracking of kinetic, potential, and total mechanical energies.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def compute_forces(positions: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """Compute pairwise Lennard-Jones forces plus external gravitational field."""
    n_atoms = positions.shape[0]
    forces = np.zeros_like(positions)
    forces[:, :] += gravity
    max_force = 1000.0  # Force clamping to prevent numerical blow-up at short range

    for i in range(n_atoms - 1):
        for j in range(i + 1, n_atoms):
            dr = positions[i] - positions[j]
            r2 = max(float(dr.dot(dr)), 0.01)
            inv_r6 = r2**-3
            f_mag = min((48.0 / r2) * (inv_r6 - 0.5) * inv_r6, max_force)
            f_ij = f_mag * dr
            forces[i] += f_ij
            forces[j] -= f_ij

    return forces


def kinetic_energy(velocities: np.ndarray) -> float:
    """Compute total kinetic energy (unit mass m = 1)."""
    return 0.5 * float(np.sum(velocities**2))


def potential_energy(positions: np.ndarray, gravity: np.ndarray) -> float:
    """Compute total Lennard-Jones + gravitational potential energy."""
    n_atoms = positions.shape[0]
    e_pot = 0.0

    for i in range(n_atoms - 1):
        for j in range(i + 1, n_atoms):
            dr = positions[i] - positions[j]
            r2 = float(np.sum(dr**2))
            if r2 > 0.0:
                inv_r6 = (1.0 / r2) ** 3
                e_pot += 4.0 * (inv_r6**2 - inv_r6)

    # Gravitational potential energy U_g = - m * (g . r)
    e_pot -= float(np.sum(positions @ gravity))
    return e_pot


def velocity_verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    dt: float,
    n_steps: int,
    box_limit: float,
    gravity: np.ndarray,
) -> np.ndarray:
    """Advance system by n_steps using Velocity-Verlet with elastic reflective walls."""
    dim = positions.shape[1]
    for _ in range(n_steps):
        velocities += 0.5 * dt * forces
        positions += dt * velocities

        for d in range(dim):
            hit_max = positions[:, d] > box_limit
            hit_min = positions[:, d] < -box_limit
            velocities[hit_max, d] *= -1.0
            velocities[hit_min, d] *= -1.0
            positions[hit_max, d] = box_limit
            positions[hit_min, d] = -box_limit

        forces = compute_forces(positions, gravity)
        velocities += 0.5 * dt * forces

    return forces


def run_simulation(enable_gravity: bool = True, save_gif: bool = True):
    dim = 2
    n_atoms = 64
    a0 = 1.2
    lattice_side = int(n_atoms**0.5 + 0.99)
    r_atom = 0.3
    target_temp = 2.0
    v_init = np.sqrt(target_temp)
    dt = 0.002
    n_iter = 800
    fast_steps = 5
    box_limit = (lattice_side * a0) / 2.0

    gravity = np.array([0.0, -5.0 if enable_gravity else 0.0])

    # Initialize square lattice positions and zero-momentum Maxwell-Boltzmann velocities
    positions = np.zeros((n_atoms, dim))
    for i in range(n_atoms):
        for k in range(dim):
            coord = (i // lattice_side**k) % lattice_side
            positions[i, k] = (coord - (lattice_side - 1) * 0.5) * a0

    velocities = v_init * np.random.standard_normal((n_atoms, dim))
    velocities -= np.mean(velocities, axis=0)
    forces = compute_forces(positions, gravity)

    time_hist = []
    energy_hist = np.empty((0, 3))

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(8, 9))
    gs = plt.GridSpec(3, 1)

    ax = fig.add_subplot(
        gs[:2, 0],
        xlim=(-box_limit * 1.1, box_limit * 1.1),
        ylim=(-box_limit * 1.1, box_limit * 1.1),
        aspect="equal",
    )
    ax.add_patch(
        plt.Rectangle(
            (-box_limit, -box_limit),
            2 * box_limit,
            2 * box_limit,
            linewidth=2,
            edgecolor="white",
            facecolor="none",
        )
    )
    ax.set_title("2D Lennard-Jones Molecular Dynamics (Velocity-Verlet)")
    atoms_plot = ax.plot(
        [], [], marker="o", color="tomato", linestyle="None", ms=r_atom * 250 / box_limit
    )

    y_min_e = -5 if enable_gravity else -3
    ax_e = fig.add_subplot(gs[-1, 0], xlim=(0, n_iter * dt), ylim=(y_min_e, 15))
    energy_curves = ax_e.plot([], [], [], [], [], [])
    ax_e.set_xlabel("Time $t$")
    ax_e.set_ylabel("Energy per atom")
    ax_e.legend(["Kinetic $E_k$", "Potential $E_p$", "Total $E_m$"], loc="upper right")
    plt.tight_layout()

    def animate(frame: int):
        nonlocal forces, energy_hist
        if frame > 0:
            forces = velocity_verlet_step(
                positions, velocities, forces, dt, fast_steps, box_limit, gravity
            )
        for a in range(n_atoms):
            atoms_plot[a].set_data([positions[a, 0]], [positions[a, 1]])

        t_curr = frame * fast_steps * dt
        time_hist.append(t_curr)
        ek = kinetic_energy(velocities) / n_atoms
        ep = potential_energy(positions, gravity) / n_atoms
        energy_hist = np.append(energy_hist, [[ek, ep, ek + ep]], axis=0)

        for k in range(3):
            energy_curves[k].set_data(time_hist, energy_hist[:, k])

        return atoms_plot + energy_curves

    n_frames = n_iter // fast_steps + 1
    ani = animation.FuncAnimation(
        fig, animate, frames=n_frames, blit=True, interval=1, repeat=False
    )

    os.makedirs("figures", exist_ok=True)
    if save_gif:
        print("Saving animation to figures/lennard_jones_2d.gif ...")
        ani.save("figures/lennard_jones_2d.gif", writer=animation.PillowWriter(fps=30))

    plt.savefig("figures/lennard_jones_2d_final.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    run_simulation(enable_gravity=True, save_gif=True)
