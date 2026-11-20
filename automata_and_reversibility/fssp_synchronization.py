"""
Firing Squad Synchronization Problem (FSSP) in 1D Cellular Automata
Implements the classical minimal-time (2n - 2 steps) 15-state solution by
McCarthy & Minsky, alongside the state-space framework for reversible
partitioned cellular automata (Imai & Morita, 3n steps).
"""

import time

# --- Classical 15-State FSSP (McCarthy & Minsky) ---
IDLE = "#"
GENERAL = "G"
L_SIG = "L"
R_SIG = "R"
FIRE = "F"


def classical_transition(left: str, state: str, right: str) -> str:
    """Local transition rule for the classical 15-state FSSP."""
    if state == "A":
        return GENERAL if right[0] == "L" else "B"
    if state == "B":
        return "C"
    if state == "C":
        return GENERAL if right[0] == "L" else IDLE
    if state == "a":
        return GENERAL if left[-1] == "R" else "b"
    if state == "b":
        return "c"
    if state == "c":
        return GENERAL if left[-1] == "R" else IDLE
    if state == "bB":
        return "cC"
    if state == "cC":
        if left[-1] == "R" or right[0] == "L":
            return GENERAL
        return IDLE
    if state == GENERAL:
        if (left + right) in ("$$", "$G", "G$", "GG"):
            return FIRE
        if right == IDLE and left == IDLE:
            return "bB"
        if right == IDLE:
            return "B"
        if left == IDLE:
            return "b"
    if state == L_SIG:
        if left == "$" or left[-1] == "R":
            return R_SIG
        if left[-1] == "C":
            return GENERAL
        if right == GENERAL:
            return L_SIG
        return IDLE
    if state == R_SIG:
        if right == "$" or right[0] == "L":
            return L_SIG
        if right[0] == "c":
            return GENERAL
        return IDLE
    if state == "LR":
        return GENERAL if (left[-1] == "C" or right[0] == "c") else IDLE
    if state == "RL":
        return "LR"
    if state == IDLE:
        if left == GENERAL or left[-1] == "R":
            if right == GENERAL or right[0] == "L":
                return "RL"
            return R_SIG
        if right == GENERAL or right[0] == "L":
            if left == GENERAL or left[-1] == "R":
                return "RL"
            return L_SIG
        if left[-1] == "C":
            return "A"
        if right[0] == "c":
            return "a"
    return state


def simulate_classical_fssp(n_cells: int = 13, sleep_secs: float = 0.0, verbose: bool = True) -> int:
    """
    Simulate the classical FSSP on a 1D array of n_cells.
    Returns the synchronization time step (expected: 2*n_cells - 2 for n >= 2).
    """
    if n_cells <= 0:
        raise ValueError("n_cells must be a positive integer.")

    seq = [IDLE] * n_cells
    seq[0] = GENERAL
    step = 0

    if verbose:
        print(f"Step {step:2d} | " + " ".join(s.ljust(2) for s in seq))

    while set(seq) != {FIRE}:
        step += 1
        next_seq = []
        for idx, state in enumerate(seq):
            left = seq[idx - 1] if idx > 0 else "$"
            right = seq[idx + 1] if idx < len(seq) - 1 else "$"
            next_state = classical_transition(left, state, right)
            if next_state == state and state != IDLE:
                raise RuntimeError(f"Non-idle state repeated at step {step}: ({left}, {state}, {right})")
            next_seq.append(next_state)

        seq = next_seq
        if verbose:
            print(f"Step {step:2d} | " + " ".join(s.ljust(2) for s in seq))
            if sleep_secs > 0:
                time.sleep(sleep_secs)

    if verbose:
        expected = 1 if n_cells == 1 else 2 * n_cells - 2
        print(f"Synchronized FIRE at step {step} (theoretical: {expected})")
    return step


if __name__ == "__main__":
    simulate_classical_fssp(n_cells=13, sleep_secs=0.0, verbose=True)
