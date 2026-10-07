"""Occupancy-measure decision-focused learning for finite MDPs.

Formulas follow arXiv:2610.08384v1. The discount factor is beta. The initial
distribution is gamma. That is the paper's convention, not the usual gamma-discount
notation.
"""

import jax

jax.config.update("jax_enable_x64", True)

from dfl_occupancy.aggregate import (
    aggregate_mdp,
    aggregated_flow_matrix,
    lift_occupancy,
    softmax_membership,
)
from dfl_occupancy.occupancy import (
    OccupancySolution,
    fixed_basis_loss_gradient,
    flow_matrix,
    implicit_basis_jacobian,
    jax_flow_matrix,
    policy_from_occupancy,
    solve_occupancy_lp,
    transition_kernel,
)
from dfl_occupancy.sketch import (
    gaussian_row_sketches,
    inference_occupancy,
    sketched_constraint_matrix,
    solve_sketched_lp,
    training_occupancy,
)
from dfl_occupancy.surrogate import (
    augmented_lagrangian,
    augmented_lagrangian_grad_y,
    surrogate_gap,
)
from dfl_occupancy.toy import ToyMDP, make_toy_mdp, same_basis_perturbation

__all__ = [
    "OccupancySolution",
    "ToyMDP",
    "aggregate_mdp",
    "aggregated_flow_matrix",
    "augmented_lagrangian",
    "augmented_lagrangian_grad_y",
    "fixed_basis_loss_gradient",
    "flow_matrix",
    "gaussian_row_sketches",
    "implicit_basis_jacobian",
    "inference_occupancy",
    "jax_flow_matrix",
    "lift_occupancy",
    "make_toy_mdp",
    "policy_from_occupancy",
    "same_basis_perturbation",
    "sketched_constraint_matrix",
    "softmax_membership",
    "solve_occupancy_lp",
    "solve_sketched_lp",
    "surrogate_gap",
    "training_occupancy",
    "transition_kernel",
]
