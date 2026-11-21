# Computational Physics & Numerical Simulations

A collection of self-contained Python numerical simulations spanning **quantum mechanics**, **statistical physics**, **molecular dynamics**, **stochastic processes**, **classical mechanics**, **cellular automata**, and **combinatorial optimization**.

---

## 📂 Repository Structure

```text
computational-physics-simulations/
├── quantum_mechanics/
│   └── schrodinger_1d.py               # 1D TDSE solver: Crank-Nicolson vs. Explicit Euler
├── statistical_physics/
│   ├── ising_2d.py                     # 2D Ising model phase transition (Metropolis MC)
│   └── binary_alloy_quenching.py       # 2D binary alloy spinodal decomposition & nucleation (Kawasaki MC)
├── molecular_dynamics/
│   └── lennard_jones_2d.py             # 2D Lennard-Jones fluid under gravity (Velocity-Verlet)
├── stochastic_processes/
│   └── random_walk_2d.py               # 2D Brownian diffusion: free, confined, and harmonic trap
├── classical_mechanics/
│   └── coupled_oscillators.py          # 1D coupled harmonic chain & energy conservation
├── automata_and_reversibility/
│   └── fssp_synchronization.py         # Firing Squad Synchronization Problem (1D cellular automata)
├── optimization/
│   └── tsp_simulated_annealing.py      # Traveling Salesman Problem via Simulated Annealing
├── figures/                            # Generated plots and animations
└── requirements.txt
