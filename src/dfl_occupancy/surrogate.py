"""Augmented Lagrangian surrogate, Equation (5) and Proposition 4.1.

L(y) = -r_true^T y + (H_true y - gamma)^T nu* + rho ||H_true y - gamma||^2.

The norm is the squared Euclidean norm. The paper calls this term a quadratic
penalty. Proposition 4.1 states that for every rho > 0 and every predicted
occupancy y_theta*, L(y_theta*) - L(y_true*) >= 0, with equality if and only
if y_theta* is optimal for the true LP.
"""

from __future__ import annotations

import numpy as np


def augmented_lagrangian(y, r_true, H_true, gamma, nu, rho):
    """Equation (5) evaluated at an occupancy y."""
    y, r_true, resid, nu, rho = _parts(y, r_true, H_true, gamma, nu, rho)
    return float(-r_true @ y + resid @ nu + rho * np.dot(resid, resid))


def augmented_lagrangian_grad_y(y, r_true, H_true, gamma, nu, rho):
    """Gradient of Equation (5) with respect to y. nu* and H_true are fixed."""
    y, r_true, resid, nu, rho = _parts(y, r_true, H_true, gamma, nu, rho)
    return -r_true + H_true.T @ nu + (2.0 * rho) * (H_true.T @ resid)


def surrogate_gap(y_pred, y_true, r_true, H_true, gamma, nu, rho):
    """L(y_pred) - L(y_true), the nonnegativity claim in Proposition 4.1."""
    return augmented_lagrangian(y_pred, r_true, H_true, gamma, nu, rho) - augmented_lagrangian(
        y_true, r_true, H_true, gamma, nu, rho
    )


def _parts(y, r_true, H_true, gamma, nu, rho):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    r_true = np.asarray(r_true, dtype=np.float64).reshape(-1)
    gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
    nu = np.asarray(nu, dtype=np.float64).reshape(-1)
    H_true = np.asarray(H_true, dtype=np.float64)
    rho = float(rho)
    if rho < 0.0:
        raise ValueError("rho must be nonnegative; Proposition 4.1 uses rho > 0")
    if H_true.shape != (gamma.shape[0], y.shape[0]):
        raise ValueError("H_true has an incompatible shape")
    if r_true.shape != y.shape or nu.shape != gamma.shape:
        raise ValueError("r_true, y, nu, and gamma have incompatible shapes")
    resid = H_true @ y - gamma
    return y, r_true, resid, nu, rho
