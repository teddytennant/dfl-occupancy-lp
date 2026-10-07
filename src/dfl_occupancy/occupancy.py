"""Occupancy-measure LP and the fixed-basis Jacobian.

LP-1 in arXiv:2610.08384v1:
    y_theta* = argmax_{y >= 0} r_theta^T y  subject to  H_theta y = gamma.

The flow matrix is
    [H_theta]_{s, (s', a')} = delta_{s, s'} - beta P_theta(s | s', a').

Equation (3), inside a fixed optimal basis B:
    dy_B / d theta = -(H_B)^{-1} (d H_B / d theta) y_B.

Equation (4), for a loss that depends on theta only through y:
    dL / d theta = -(grad_{y_B} L)^T (H_B)^{-1} (d H_B / d theta) y_B.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dfl_occupancy.simplex import solve_lp


@dataclass(frozen=True)
class OccupancySolution:
    """Basic optimal solution of an occupancy LP.

    ``y`` has shape (n_states, n_actions). Flat index s * n_actions + a matches
    the column (s, a) of H. ``dual`` solves H_B^T dual = r_B for the system
    that was actually solved. For the unsketched true LP this is nu* in Eq. (5).
    """

    y: np.ndarray
    y_flat: np.ndarray
    basis: np.ndarray
    dual: np.ndarray
    objective: float
    H: np.ndarray
    constraint_matrix: np.ndarray
    constraint_rhs: np.ndarray
    n_states: int
    n_actions: int


def transition_kernel(logits):
    """Row-softmax transition kernel. P[s, a, s_next] = P(s_next | s, a)."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 3 or logits.shape[0] != logits.shape[2]:
        raise ValueError("logits must have shape (n_states, n_actions, n_states)")
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def flow_matrix(P, beta):
    """Build H from a kernel P[s, a, s_next] = P(s_next | s, a)."""
    P = _as_kernel(P)
    beta = _as_beta(beta)
    n_states, n_actions, _ = P.shape
    next_state_major = np.transpose(P, (2, 0, 1)).reshape(n_states, n_states * n_actions)
    H = -beta * next_state_major
    columns = (np.arange(n_states)[:, None] * n_actions + np.arange(n_actions)[None, :]).reshape(-1)
    rows = np.repeat(np.arange(n_states), n_actions)
    H[rows, columns] += 1.0
    return H


def jax_flow_matrix(logits, beta):
    """Differentiable flow matrix, same layout as ``flow_matrix``."""
    logits = jnp.asarray(logits, dtype=jnp.float64)
    P = jax.nn.softmax(logits, axis=-1)
    n_states, n_actions, _ = P.shape
    next_state_major = jnp.transpose(P, (2, 0, 1)).reshape(n_states, n_states * n_actions)
    H = -float(beta) * next_state_major
    columns = (jnp.arange(n_states)[:, None] * n_actions + jnp.arange(n_actions)[None, :]).reshape(-1)
    rows = jnp.repeat(jnp.arange(n_states), n_actions)
    return H.at[rows, columns].add(1.0)


def solve_occupancy_lp(P, reward, gamma, beta):
    """Solve LP-1. ``reward`` has shape (n_states, n_actions)."""
    P = _as_kernel(P)
    reward = _as_reward(reward, P.shape[0], P.shape[1])
    gamma = _as_gamma(gamma, P.shape[0])
    beta = _as_beta(beta)
    H = flow_matrix(P, beta)
    result = solve_lp(reward.reshape(-1), H, gamma)
    return _pack(result, reward.shape, H, H, gamma)


def policy_from_occupancy(y):
    """pi(a | s) = y(s, a) / sum_{a'} y(s, a'), as in Section 3."""
    y = np.asarray(y, dtype=np.float64)
    mass = y.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("every state occupancy must be positive")
    return y / mass


def implicit_basis_jacobian(constraint_fn, logits, y_flat, basis):
    """Equation (3) for A(theta) y = b with b independent of theta.

    ``constraint_fn(logits)`` returns A with shape (m, n). Nonbasic rows of the
    returned Jacobian are zero. Shape is (n, *logits.shape).
    """
    basis = tuple(int(index) for index in basis)
    logits = jnp.asarray(logits, dtype=jnp.float64)
    y_flat = jnp.asarray(y_flat, dtype=jnp.float64).reshape(-1)
    basis_index = jnp.asarray(basis, dtype=jnp.int32)
    y_B = y_flat[basis_index]

    def basis_block(lg):
        return constraint_fn(lg)[:, jnp.asarray(basis, dtype=jnp.int32)]

    block = basis_block(logits)
    derivative = jax.jacrev(basis_block)(logits)
    m, width = block.shape
    contracted = jnp.einsum("mbp,b->mp", derivative.reshape(m, width, -1), y_B)
    solved = jax.scipy.linalg.solve(block, contracted)
    dy_B = (-solved).reshape((width,) + logits.shape)
    dy = jnp.zeros((y_flat.shape[0],) + logits.shape, dtype=jnp.float64)
    return dy.at[basis_index].set(dy_B)


def fixed_basis_loss_gradient(grad_y, constraint_fn, logits, y_flat, basis):
    """Equation (4): dL/d theta = (grad_y)^T dy/d theta, with dy from Eq. (3)."""
    dy = implicit_basis_jacobian(constraint_fn, logits, y_flat, basis)
    grad_y = jnp.asarray(grad_y, dtype=jnp.float64).reshape(-1)
    return jnp.einsum("j,j...->...", grad_y, dy)


def flow_constraint_fn(beta):
    """Unsketched constraint map logits -> H_theta."""

    def constraint(logits):
        return jax_flow_matrix(logits, beta)

    return constraint


def _pack(result, reward_shape, H, constraint_matrix, constraint_rhs):
    n_states, n_actions = reward_shape
    y_flat = np.asarray(result.x, dtype=np.float64)
    return OccupancySolution(
        y=y_flat.reshape(n_states, n_actions),
        y_flat=y_flat,
        basis=np.asarray(result.basis, dtype=int),
        dual=np.asarray(result.dual, dtype=np.float64),
        objective=float(result.objective),
        H=np.asarray(H, dtype=np.float64),
        constraint_matrix=np.asarray(constraint_matrix, dtype=np.float64),
        constraint_rhs=np.asarray(constraint_rhs, dtype=np.float64).reshape(-1),
        n_states=n_states,
        n_actions=n_actions,
    )


def _as_kernel(P):
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 3 or P.shape[0] != P.shape[2]:
        raise ValueError("P must have shape (n_states, n_actions, n_states)")
    if np.any(P < -1e-10):
        raise ValueError("P must be nonnegative")
    if np.max(np.abs(P.sum(axis=-1) - 1.0)) > 1e-7:
        raise ValueError("P must sum to 1 over next states")
    return P


def _as_reward(reward, n_states, n_actions):
    reward = np.asarray(reward, dtype=np.float64)
    if reward.shape != (n_states, n_actions):
        raise ValueError("reward must have shape (n_states, n_actions)")
    return reward


def _as_gamma(gamma, n_states):
    gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
    if gamma.shape != (n_states,):
        raise ValueError("gamma must have one entry per state")
    if np.any(gamma < -1e-12):
        raise ValueError("gamma must be nonnegative")
    if abs(float(gamma.sum()) - 1.0) > 1e-8:
        raise ValueError("gamma must sum to 1 so that 1^T y = 1/(1-beta)")
    return gamma


def _as_beta(beta):
    beta = float(beta)
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must lie in [0, 1)")
    return beta
