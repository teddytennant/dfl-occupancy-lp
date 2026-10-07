"""Two-phase revised simplex for max c^T x s.t. A x = b, x >= 0.

The paper identifies the active occupancy basis by a pivoting algorithm and
treats the solver as a deterministic tie-break. This implementation uses
Bland's rule. The paper does not name that rule. For these tiny LPs, a
lexicographic basis enumeration is the fallback if pivoting fails.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimplexResult:
    x: np.ndarray
    basis: np.ndarray
    dual: np.ndarray
    objective: float
    status: str


def solve_lp(c, A, b, tol=1e-9):
    """Maximize c^T x subject to A x = b and x >= 0.

    Returns a basic optimal solution. ``dual`` solves A_B^T dual = c_B, which
    is the equality dual of the maximization problem.
    """
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if A.ndim != 2 or A.shape[1] != c.shape[0] or A.shape[0] != b.shape[0]:
        raise ValueError("A, b, and c have incompatible shapes")
    status, basis, x, dual = _revised_simplex(c, A, b, tol=tol)
    if status != "optimal":
        enumerated = _enumerate_optimal_basis(c, A, b, tol=max(tol, 1e-8))
        if enumerated is None:
            raise RuntimeError(f"LP solve failed ({status})")
        objective, basis, x, dual = enumerated
        status = "optimal"
    residual = A @ x - b
    if np.linalg.norm(residual) > 1e-6:
        raise RuntimeError("LP solver returned an infeasible point")
    if np.any(x < -1e-7):
        raise RuntimeError("LP solver returned a negative point")
    x = np.maximum(x, 0.0)
    return SimplexResult(
        x=x,
        basis=np.asarray(basis, dtype=int),
        dual=np.asarray(dual, dtype=np.float64),
        objective=float(c @ x),
        status=status,
    )


def _revised_simplex(c, A, b, tol, max_iter=4000):
    m, n = A.shape
    if m == 0:
        raise ValueError("LP has no equality rows")
    row_sign = np.where(b < -tol, -1.0, 1.0)
    A_work = A * row_sign[:, None]
    b_work = b * row_sign
    eye = np.eye(m, dtype=np.float64)
    A_phase1 = np.concatenate([A_work, eye], axis=1)
    c_phase1 = np.concatenate([np.zeros(n), -np.ones(m)])
    basis = list(range(n, n + m))
    status, basis, x_full = _pivot(c_phase1, A_phase1, b_work, basis, tol, max_iter)
    if status != "optimal":
        return status, None, None, None
    if float(np.sum(x_full[n:])) > 1e-6:
        return "infeasible", None, None, None
    for _ in range(m + 8):
        artificial_rows = [i for i, col in enumerate(basis) if col >= n]
        if not artificial_rows:
            break
        B = A_phase1[:, basis]
        progressed = False
        for row in artificial_rows:
            try:
                tableau_row = np.linalg.solve(B.T, np.eye(m)[row])
            except np.linalg.LinAlgError:
                return "singular", None, None, None
            candidates = [
                j
                for j in range(n)
                if j not in basis and abs(float(tableau_row @ A_phase1[:, j])) > 1e-8
            ]
            if not candidates:
                continue
            basis[row] = min(candidates)
            progressed = True
            break
        if not progressed:
            return "degenerate_artificial", None, None, None
    if any(col >= n for col in basis):
        return "degenerate_artificial", None, None, None
    status, basis, x = _pivot(c, A_work, b_work, basis, tol, max_iter)
    if status != "optimal":
        return status, None, None, None
    basis = np.array(sorted(basis), dtype=int)
    try:
        x_B = np.linalg.solve(A[:, basis], b)
        dual = np.linalg.solve(A[:, basis].T, c[basis])
    except np.linalg.LinAlgError:
        return "singular", None, None, None
    if np.any(x_B < -1e-6):
        return "numerical", None, None, None
    x_out = np.zeros(n, dtype=np.float64)
    x_out[basis] = np.maximum(x_B, 0.0)
    reduced = c - A.T @ dual
    reduced[basis] = 0.0
    if np.any(reduced > 1e-6):
        return "not_optimal", None, None, None
    return "optimal", basis, x_out, dual


def _pivot(c, A, b, basis, tol, max_iter):
    m, n = A.shape
    basis = list(basis)
    seen = {}
    for _ in range(max_iter):
        key = tuple(basis)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 6:
            return "cycle", basis, None
        B = A[:, basis]
        try:
            x_B = np.linalg.solve(B, b)
            dual = np.linalg.solve(B.T, c[list(basis)])
        except np.linalg.LinAlgError:
            return "singular", basis, None
        if np.any(x_B < -1e-5):
            return "infeasible_basis", basis, None
        x_B = np.where(x_B < 0.0, 0.0, x_B)
        reduced = c - A.T @ dual
        candidates = [j for j in range(n) if j not in basis and reduced[j] > tol]
        if not candidates:
            x = np.zeros(n, dtype=np.float64)
            x[basis] = x_B
            return "optimal", basis, x
        enter = min(candidates)
        try:
            direction = np.linalg.solve(B, A[:, enter])
        except np.linalg.LinAlgError:
            return "singular", basis, None
        positive = [i for i in range(m) if direction[i] > tol]
        if not positive:
            return "unbounded", basis, None
        ratios = [(x_B[i] / direction[i], basis[i], i) for i in positive]
        min_ratio = min(item[0] for item in ratios)
        tied = [
            item
            for item in ratios
            if abs(item[0] - min_ratio) <= 1e-8 * (1.0 + abs(min_ratio)) + 1e-12
        ]
        _, _, leave_local = min(tied, key=lambda item: item[1])
        basis[leave_local] = enter
    return "max_iter", basis, None


def _enumerate_optimal_basis(c, A, b, tol):
    m, n = A.shape
    if n > 12 or m > 6 or m > n:
        return None
    best = None
    for cols in itertools.combinations(range(n), m):
        B = A[:, cols]
        sign, logdet = np.linalg.slogdet(B)
        if abs(sign) < 0.5 or logdet < -12.0:
            continue
        try:
            x_B = np.linalg.solve(B, b)
            dual = np.linalg.solve(B.T, c[list(cols)])
        except np.linalg.LinAlgError:
            continue
        if np.any(x_B < -tol):
            continue
        x_B = np.maximum(x_B, 0.0)
        reduced = c - A.T @ dual
        if any(reduced[j] > tol for j in range(n) if j not in cols):
            continue
        objective = float(c[list(cols)] @ x_B)
        basis = tuple(cols)
        if best is None or objective > best[0] + tol or (
            abs(objective - best[0]) <= tol and basis < best[1]
        ):
            x = np.zeros(n, dtype=np.float64)
            x[list(cols)] = x_B
            best = (objective, np.array(basis, dtype=int), x, dual)
    return best
