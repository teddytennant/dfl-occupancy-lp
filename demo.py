"""Print the toy occupancy, the surrogate gap, and the Jacobian check."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dfl_occupancy.occupancy import (
    implicit_basis_jacobian,
    flow_constraint_fn,
    solve_occupancy_lp,
    transition_kernel,
)
from dfl_occupancy.surrogate import surrogate_gap
from dfl_occupancy.toy import make_toy_mdp, same_basis_perturbation


def main():
    toy = make_toy_mdp(0)
    rho = 1.0
    eps = 1e-5
    true = solve_occupancy_lp(
        transition_kernel(toy.logits), toy.reward, toy.gamma, toy.beta
    )
    perturbed_logits = same_basis_perturbation(toy.logits)
    perturbed = solve_occupancy_lp(
        transition_kernel(perturbed_logits), toy.reward, toy.gamma, toy.beta
    )
    gap = surrogate_gap(
        perturbed.y, true.y, toy.reward, true.H, toy.gamma, true.dual, rho
    )
    analytic = np.asarray(
        implicit_basis_jacobian(
            flow_constraint_fn(toy.beta),
            toy.logits,
            true.y_flat,
            true.basis,
        )
    )
    finite = np.zeros_like(analytic)
    for idx in np.ndindex(toy.logits.shape):
        plus = toy.logits.copy()
        minus = toy.logits.copy()
        plus[idx] += eps
        minus[idx] -= eps
        plus_y = solve_occupancy_lp(
            transition_kernel(plus), toy.reward, toy.gamma, toy.beta
        ).y_flat
        minus_y = solve_occupancy_lp(
            transition_kernel(minus), toy.reward, toy.gamma, toy.beta
        ).y_flat
        finite[(slice(None),) + idx] = (plus_y - minus_y) / (2.0 * eps)
    relative_error = np.linalg.norm(analytic - finite) / np.linalg.norm(finite)
    np.set_printoptions(precision=8, suppress=True, floatmode="fixed")
    print("true occupancy")
    print(true.y)
    print("perturbed occupancy")
    print(perturbed.y)
    print(f"surrogate gap {gap:.8e}")
    print(f"finite-difference vs analytic gradient relative error {relative_error:.8e}")


if __name__ == "__main__":
    main()
