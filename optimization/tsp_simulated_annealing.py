"""
Traveling Salesman Problem (TSP) via Simulated Annealing / Metropolis Monte Carlo
Finds near-optimal closed tours over N random 2D cities using pairwise swap moves
and a cooling schedule T(step), compared against the Beardwood–Halton–Hammersley
asymptotic length estimate L ≈ 0.7124 * sqrt(N).
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def tour_length(points: np.ndarray) -> float:
    """Compute total closed Euclidean tour length."""
    shifted = np.roll(points, -1, axis=0)
    return float(np.sum(np.linalg.norm(points - shifted, axis=1)))


def simulated_annealing_tsp(
    points: np.ndarray,
    n_steps: int = 8000,
    T_init: float = 0.5,
    T_final: float = 0.002,
) -> tuple[np.ndarray, list[float]]:
    """Minimize closed tour length using Metropolis swap moves with exponential cooling."""
    current_tour = points.copy()
    current_len = tour_length(current_tour)
    best_tour = current_tour.copy()
    best_len = current_len
    n_cities = len(points)

    temperatures = np.geomspace(T_init, T_final, n_steps)
    length_history = []

    for T in temperatures:
        i, j = np.random.choice(n_cities, size=2, replace=False)
        candidate = current_tour.copy()
        candidate[i], candidate[j] = candidate[j].copy(), candidate[i].copy()
        candidate_len = tour_length(candidate)

        dL = candidate_len - current_len
        if dL <= 0.0 or np.random.random() < np.exp(-dL / T):
            current_tour = candidate
            current_len = candidate_len
            if current_len < best_len:
                best_tour = current_tour.copy()
                best_len = current_len

        length_history.append(current_len)

    return best_tour, length_history


def main():
    np.random.seed(42)
    n_cities = 25
    cities = np.random.rand(n_cities, 2)

    initial_len = tour_length(cities)
    bhh_estimate = 0.7124 * np.sqrt(n_cities)  # Theoretical asymptotic constant in [0,1]^2

    best_tour, history = simulated_annealing_tsp(cities, n_steps=10000)
    final_len = tour_length(best_tour)

    print(f"Initial tour length : {initial_len:.4f}")
    print(f"Optimized length    : {final_len:.4f}")
    print(f"BHH asymptotic est. : {bhh_estimate:.4f}")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # 1. Initial tour
    closed_init = np.vstack([cities, cities[0]])
    axes[0].plot(closed_init[:, 0], closed_init[:, 1], "o-", color="tab:gray")
    axes[0].set_title(f"Initial Random Tour ($L = {initial_len:.2f}$)")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Optimized tour
    closed_best = np.vstack([best_tour, best_tour[0]])
    axes[1].plot(closed_best[:, 0], closed_best[:, 1], "o-", color="tab:green")
    axes[1].set_title(f"Optimized Tour ($L = {final_len:.2f}$)")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$y$")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # 3. Convergence curve
    axes[2].plot(history, color="tab:blue", linewidth=1.2, label="Current tour length")
    axes[2].axhline(
        bhh_estimate, color="tab:red", linestyle="--", label=f"BHH bound ($\\approx {bhh_estimate:.2f}$)"
    )
    axes[2].set_title("Simulated Annealing Convergence")
    axes[2].set_xlabel("Monte Carlo Step")
    axes[2].set_ylabel("Tour Length $L$")
    axes[2].legend()
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/tsp_simulated_annealing.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
