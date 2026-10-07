"""Gaussian row sketches of the occupancy LP, Equation (6).

Draw S in R^{k x |S|} with i.i.d. entries N(0, 1/k), k = alpha |S|, alpha in
(0, 1]. When k < |S|, keep the mass row 1^T y = 1/(1-beta). When k = |S|,
the sketch is invertible almost surely and the mass row is omitted.

Training uses the average of K sketched solutions. Inference solves the
unsketched predicted LP.
"""

from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dfl_occupancy.occupancy import (
    _as_beta,
    _as_gamma,
    _as_kernel,
    _as_reward,
    _pack,
    flow_matrix,
    solve_occupancy_lp,
)
from dfl_occupancy.simplex import solve_lp


def gaussian_row_sketches(key, n_sketches, n_rows, n_states):
    """K sketches with entries N(0, 1/k), as stated before Equation (6)."""
    n_sketches = int(n_sketches)
    n_rows = int(n_rows)
    n_states = int(n_states)
    if n_sketches < 1 or n_rows < 1 or n_rows > n_states:
        raise ValueError("need K >= 1 and 1 <= k <= |S|")
    if not isinstance(key, jax.Array):
        key = jax.random.PRNGKey(int(key))
    draws = jax.random.normal(key, (n_sketches, n_rows, n_states), dtype=jnp.float64)
    return draws / jnp.sqrt(n_rows)


def sketched_constraint_matrix(H, gamma, beta, sketch):
    """Equality system in Equation (6), including the k = |S| branch."""
    H = np.asarray(H, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
    beta = _as_beta(beta)
    sketch = np.asarray(sketch, dtype=np.float64)
    if sketch.ndim != 2 or sketch.shape[1] != H.shape[0] or gamma.shape[0] != H.shape[0]:
        raise ValueError("sketch must have shape (k, n_states)")
    k, n_states = sketch.shape
    if k < n_states:
        matrix = np.vstack([sketch @ H, np.ones((1, H.shape[1]), dtype=np.float64)])
        rhs = np.concatenate([sketch @ gamma, np.array([1.0 / (1.0 - beta)])])
        return matrix, rhs
    if k == n_states:
        return sketch @ H, sketch @ gamma
    raise ValueError("k exceeds |S|; Equation (6) takes alpha in (0, 1]")


def solve_sketched_lp(P, reward, gamma, beta, sketch):
    """Solve one sketched occupancy LP. The returned dual is not nu*."""
    P = _as_kernel(P)
    reward = _as_reward(reward, P.shape[0], P.shape[1])
    gamma = _as_gamma(gamma, P.shape[0])
    beta = _as_beta(beta)
    H = flow_matrix(P, beta)
    matrix, rhs = sketched_constraint_matrix(H, gamma, beta, sketch)
    result = solve_lp(reward.reshape(-1), matrix, rhs)
    return _pack(result, reward.shape, H, matrix, rhs)


def training_occupancy(P, reward, gamma, beta, sketches):
    """Average of the K sketched solutions used during training."""
    sketches = np.asarray(sketches, dtype=np.float64)
    if sketches.ndim != 3:
        raise ValueError("sketches must have shape (K, k, n_states)")
    solutions = [solve_sketched_lp(P, reward, gamma, beta, sketch).y for sketch in sketches]
    return np.mean(np.stack(solutions, axis=0), axis=0)


def inference_occupancy(P, reward, gamma, beta):
    """Unsketched predicted LP used at inference."""
    return solve_occupancy_lp(P, reward, gamma, beta).y


def jax_sketched_matrix(logits, beta, sketch):
    """Differentiable sketched constraint matrix, matching Equation (6)."""
    from dfl_occupancy.occupancy import jax_flow_matrix

    H = jax_flow_matrix(logits, beta)
    sketch = jnp.asarray(sketch, dtype=jnp.float64)
    k, n_states = sketch.shape
    sketched = sketch @ H
    if k < n_states:
        mass = jnp.ones((1, H.shape[1]), dtype=jnp.float64)
        return jnp.vstack([sketched, mass])
    return sketched
