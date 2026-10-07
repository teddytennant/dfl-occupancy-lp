"""Soft state aggregation for finite MDPs, Equations (7) and (8).

W has simplex rows. Cluster mass is mu_m = sum_s W_sm. Aggregated reward,
transitions, and initial distribution follow Equation (8). The lift is
y_disagg(s, a) = sum_m W_sm w(m, a) with y_hat(m, a) = mu_m w(m, a).

Section 5.2 continuous feature networks are not implemented.
"""

from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dfl_occupancy.occupancy import flow_matrix


def softmax_membership(logits):
    """Softmax membership. Rows lie on the simplex, as required for Equation (7)."""
    logits = jnp.asarray(logits, dtype=jnp.float64)
    if logits.ndim != 2:
        raise ValueError("membership logits must have shape (n_states, n_clusters)")
    return jax.nn.softmax(logits, axis=-1)


def aggregate_mdp(P, reward, gamma, membership):
    """Equation (8). Returns r_hat, P_hat, gamma_hat, mu."""
    P = np.asarray(P, dtype=np.float64)
    reward = np.asarray(reward, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
    W = np.asarray(membership, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != P.shape[0]:
        raise ValueError("membership must have shape (n_states, n_clusters)")
    if np.any(W < -1e-10):
        raise ValueError("membership weights must be nonnegative")
    if np.max(np.abs(W.sum(axis=1) - 1.0)) > 1e-6:
        raise ValueError("membership rows must sum to 1")
    mu = W.sum(axis=0)
    if np.any(mu <= 1e-12):
        raise ValueError("every cluster mass must be positive")
    reward_hat = (W.T @ reward) / mu[:, None]
    mixed = np.einsum("saj,jn->san", P, W)
    numerator = np.einsum("sm,san->man", W, mixed)
    P_hat = numerator / mu[:, None, None]
    gamma_hat = W.T @ gamma
    return reward_hat, P_hat, gamma_hat, mu


def aggregated_flow_matrix(P_hat, beta):
    """H_hat_{i, (m, a')} = delta_{i, m} - beta P_hat(i | m, a')."""
    return flow_matrix(P_hat, beta)


def lift_occupancy(y_hat, membership, mu):
    """Lift y_hat(m, a) = mu_m w(m, a) by y(s, a) = sum_m W_sm w(m, a)."""
    y_hat = np.asarray(y_hat, dtype=np.float64)
    W = np.asarray(membership, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    if np.any(mu <= 0.0):
        raise ValueError("cluster masses must be positive")
    weights = y_hat / mu[:, None]
    return W @ weights
