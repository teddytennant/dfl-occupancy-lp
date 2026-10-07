"""CPU checks for the occupancy LP, Equation (3), Proposition 4.1, and Equation (6)."""

import numpy as np
import pytest

from dfl_occupancy.aggregate import (
    aggregate_mdp,
    lift_occupancy,
    softmax_membership,
)
from dfl_occupancy.occupancy import (
    fixed_basis_loss_gradient,
    flow_constraint_fn,
    flow_matrix,
    jax_flow_matrix,
    policy_from_occupancy,
    solve_occupancy_lp,
    transition_kernel,
)
from dfl_occupancy.simplex import solve_lp
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
from dfl_occupancy.toy import make_toy_mdp, same_basis_perturbation
from dfl_occupancy.train import train_step


EPS = 1e-5
RHO = 1.0


def _toy_solution(logits=None):
    toy = make_toy_mdp(0)
    if logits is None:
        logits = toy.logits
    P = transition_kernel(logits)
    solution = solve_occupancy_lp(P, toy.reward, toy.gamma, toy.beta)
    return toy, logits, P, solution


def _value_iteration(P, reward, beta, tol=1e-12):
    value = np.zeros(reward.shape[0], dtype=np.float64)
    for _ in range(100000):
        action_values = reward + beta * np.einsum("saj,j->sa", P, value)
        updated = action_values.max(axis=-1)
        if np.max(np.abs(updated - value)) < tol:
            value = updated
            break
        value = updated
    action_values = reward + beta * np.einsum("saj,j->sa", P, value)
    return value, action_values.argmax(axis=-1)


def test_flow_matrix_matches_the_paper_definition():
    P = np.zeros((2, 2, 2), dtype=np.float64)
    P[0, 0] = [0.2, 0.8]
    P[0, 1] = [0.0, 1.0]
    P[1, 0] = [1.0, 0.0]
    P[1, 1] = [0.4, 0.6]
    beta = 0.5
    H = flow_matrix(P, beta)
    for state in range(2):
        for action in range(2):
            for row in range(2):
                expected = (1.0 if row == state else 0.0) - beta * P[state, action, row]
                assert H[row, state * 2 + action] == pytest.approx(expected)
    ones = np.ones(2)
    assert np.allclose(ones @ H, (1.0 - beta) * np.ones(H.shape[1]))


def test_simplex_matches_a_hand_solved_vertex():
    # max x + 2y subject to x + y = 1, x, y >= 0. Optimum is (0, 1).
    result = solve_lp(np.array([1.0, 2.0]), np.array([[1.0, 1.0]]), np.array([1.0]))
    assert result.objective == pytest.approx(2.0)
    assert result.x == pytest.approx(np.array([0.0, 1.0]))
    assert result.dual == pytest.approx(np.array([2.0]))


def test_occupancy_lp_mass_flow_and_value_iteration():
    toy, logits, P, solution = _toy_solution()
    beta = toy.beta
    assert np.all(solution.y >= -1e-10)
    assert solution.y.sum() == pytest.approx(1.0 / (1.0 - beta))
    assert np.allclose(solution.H @ solution.y_flat, toy.gamma, atol=1e-8)
    ones = np.ones(toy.n_states)
    assert np.allclose(ones @ solution.H, (1.0 - beta) * np.ones(solution.H.shape[1]))
    value, policy = _value_iteration(P, toy.reward, beta)
    assert toy.gamma @ value == pytest.approx(solution.objective, abs=1e-6)
    assert np.array_equal(policy, np.array([1, 1, 1]))
    recovered = policy_from_occupancy(solution.y).argmax(axis=-1)
    assert np.array_equal(recovered, policy)
    # Dual of the maximization LP: H_B^T nu = r_B, and H^T nu >= r.
    reduced = solution.H.T @ solution.dual - toy.reward.reshape(-1)
    assert np.all(reduced >= -1e-7)
    assert toy.gamma @ solution.dual == pytest.approx(solution.objective, abs=1e-7)
    assert np.allclose(jax_flow_matrix(logits, beta), solution.H, atol=1e-12)


def test_fixed_basis_jacobian_matches_finite_difference():
    from dfl_occupancy.occupancy import implicit_basis_jacobian

    toy, logits, _, solution = _toy_solution()
    analytic = np.asarray(
        implicit_basis_jacobian(
            flow_constraint_fn(toy.beta),
            logits,
            solution.y_flat,
            solution.basis,
        )
    )
    fd = np.zeros_like(analytic)
    for idx in np.ndindex(logits.shape):
        plus = logits.copy()
        minus = logits.copy()
        plus[idx] += EPS
        minus[idx] -= EPS
        plus_sol = solve_occupancy_lp(
            transition_kernel(plus), toy.reward, toy.gamma, toy.beta
        )
        minus_sol = solve_occupancy_lp(
            transition_kernel(minus), toy.reward, toy.gamma, toy.beta
        )
        assert set(plus_sol.basis.tolist()) == set(solution.basis.tolist())
        assert set(minus_sol.basis.tolist()) == set(solution.basis.tolist())
        fd[(slice(None),) + idx] = (plus_sol.y_flat - minus_sol.y_flat) / (2.0 * EPS)
    denom = np.linalg.norm(fd)
    assert denom > 1e-6
    relative = np.linalg.norm(analytic - fd) / denom
    flipped = np.linalg.norm(-analytic - fd) / denom
    assert relative < 1e-3
    assert flipped > 0.5


def test_decision_and_surrogate_gradients_match_finite_difference():
    toy, logits, _, true = _toy_solution()
    perturbed = same_basis_perturbation(logits)
    pred = solve_occupancy_lp(
        transition_kernel(perturbed), toy.reward, toy.gamma, toy.beta
    )
    assert set(pred.basis.tolist()) == set(true.basis.tolist())
    r_flat = toy.reward.reshape(-1)

    def decision_loss(point):
        y = solve_occupancy_lp(
            transition_kernel(point), toy.reward, toy.gamma, toy.beta
        ).y_flat
        return -r_flat @ y

    def surrogate(point):
        y = solve_occupancy_lp(
            transition_kernel(point), toy.reward, toy.gamma, toy.beta
        ).y_flat
        return augmented_lagrangian(y, r_flat, true.H, toy.gamma, true.dual, RHO)

    decision_grad = np.asarray(
        fixed_basis_loss_gradient(
            -r_flat,
            flow_constraint_fn(toy.beta),
            perturbed,
            pred.y_flat,
            pred.basis,
        )
    )
    surrogate_grad = np.asarray(
        fixed_basis_loss_gradient(
            augmented_lagrangian_grad_y(
                pred.y_flat, r_flat, true.H, toy.gamma, true.dual, RHO
            ),
            flow_constraint_fn(toy.beta),
            perturbed,
            pred.y_flat,
            pred.basis,
        )
    )
    decision_fd = np.zeros_like(perturbed)
    surrogate_fd = np.zeros_like(perturbed)
    for idx in np.ndindex(perturbed.shape):
        plus = perturbed.copy()
        minus = perturbed.copy()
        plus[idx] += EPS
        minus[idx] -= EPS
        for point in (plus, minus):
            other = solve_occupancy_lp(
                transition_kernel(point), toy.reward, toy.gamma, toy.beta
            )
            assert set(other.basis.tolist()) == set(pred.basis.tolist())
        decision_fd[idx] = (decision_loss(plus) - decision_loss(minus)) / (2.0 * EPS)
        surrogate_fd[idx] = (surrogate(plus) - surrogate(minus)) / (2.0 * EPS)
    decision_rel = np.linalg.norm(decision_grad - decision_fd) / np.linalg.norm(decision_fd)
    surrogate_rel = np.linalg.norm(surrogate_grad - surrogate_fd) / np.linalg.norm(surrogate_fd)
    assert decision_rel < 1e-3
    assert surrogate_rel < 1e-3
    assert np.linalg.norm(-decision_grad - decision_fd) / np.linalg.norm(decision_fd) > 0.5
    assert np.linalg.norm(-surrogate_grad - surrogate_fd) / np.linalg.norm(surrogate_fd) > 0.5


def test_proposition_4_1_nonnegative_surrogate_gap():
    toy, logits, P, true = _toy_solution()
    perturbed = same_basis_perturbation(logits)
    pred = solve_occupancy_lp(transition_kernel(perturbed), toy.reward, toy.gamma, toy.beta)
    assert np.linalg.norm(pred.y - true.y) > 1e-3
    for rho in (RHO, 2.5):
        gap_true = surrogate_gap(
            true.y, true.y, toy.reward, true.H, toy.gamma, true.dual, rho
        )
        gap_pred = surrogate_gap(
            pred.y, true.y, toy.reward, true.H, toy.gamma, true.dual, rho
        )
        assert gap_true == pytest.approx(0.0, abs=1e-8)
        assert gap_pred > 1e-3
    # Same transitions, worse reward: the occupancy is feasible for the true
    # flow and not optimal for r_true, so the gap is the decision regret.
    reward_bad = np.zeros_like(toy.reward)
    reward_bad[:, 0] = 1.0
    bad = solve_occupancy_lp(P, reward_bad, toy.gamma, toy.beta)
    gap_bad = surrogate_gap(bad.y, true.y, toy.reward, true.H, toy.gamma, true.dual, RHO)
    regret = toy.reward.reshape(-1) @ (true.y_flat - bad.y_flat)
    assert gap_bad > 1e-3
    assert gap_bad == pytest.approx(regret, abs=1e-6)
    assert np.linalg.norm(true.H @ bad.y_flat - toy.gamma) < 1e-7


def test_training_averages_sketches_and_inference_does_not(monkeypatch):
    toy, logits, P, unsketched = _toy_solution()
    sketches = np.random.default_rng(2).normal(size=(5, 1, toy.n_states))
    sketches = sketches / np.sqrt(1.0)
    individuals = [
        solve_sketched_lp(P, toy.reward, toy.gamma, toy.beta, sketch) for sketch in sketches
    ]
    for solution, sketch in zip(individuals, sketches):
        matrix, rhs = sketched_constraint_matrix(unsketched.H, toy.gamma, toy.beta, sketch)
        assert matrix.shape == (2, toy.n_states * toy.n_actions)
        assert rhs[-1] == pytest.approx(1.0 / (1.0 - toy.beta))
        assert np.allclose(matrix @ solution.y_flat, rhs, atol=1e-7)
        assert solution.y_flat.sum() == pytest.approx(1.0 / (1.0 - toy.beta))
        assert np.all(solution.y >= -1e-8)
    averaged = np.mean([solution.y for solution in individuals], axis=0)
    trained = training_occupancy(P, toy.reward, toy.gamma, toy.beta, sketches)
    inferred = inference_occupancy(P, toy.reward, toy.gamma, toy.beta)
    assert np.allclose(trained, averaged, atol=1e-10)
    assert np.linalg.norm(trained - inferred) > 1e-2
    # A single sketched vertex is not the training average, so replacing the
    # average by one sketch fails the comparison above.
    assert min(np.linalg.norm(solution.y - averaged) for solution in individuals) > 1e-4
    assert np.allclose(inferred, unsketched.y)

    def boom(*args, **kwargs):
        raise AssertionError("inference path sketched the LP")

    monkeypatch.setattr("dfl_occupancy.sketch.solve_sketched_lp", boom)
    again = inference_occupancy(P, toy.reward, toy.gamma, toy.beta)
    assert np.allclose(again, unsketched.y)


def test_square_sketch_omits_the_mass_row_and_matches_the_unsketched_lp():
    toy, _, P, unsketched = _toy_solution()
    factor, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    matrix, rhs = sketched_constraint_matrix(unsketched.H, toy.gamma, toy.beta, factor)
    assert matrix.shape[0] == toy.n_states
    assert rhs.shape == (toy.n_states,)
    sketched = solve_sketched_lp(P, toy.reward, toy.gamma, toy.beta, factor)
    assert np.allclose(sketched.y, unsketched.y, atol=1e-6)


def test_gaussian_sketches_have_the_paper_variance():
    import jax

    draws = np.asarray(gaussian_row_sketches(jax.random.PRNGKey(0), 4000, 2, 3))
    assert draws.shape == (4000, 2, 3)
    assert abs(float(np.var(draws)) - 0.5) < 0.05


def test_train_step_uses_the_sketch_average():
    toy, logits, P, true = _toy_solution()
    start = same_basis_perturbation(logits)
    sketches = np.random.default_rng(2).normal(size=(4, 1, toy.n_states))
    _, _, info = train_step(
        start,
        toy.reward,
        toy.gamma,
        toy.beta,
        sketches,
        toy.reward,
        true.H,
        true.dual,
        rho=RHO,
        lr=1e-3,
    )
    trained = training_occupancy(P_from(start), toy.reward, toy.gamma, toy.beta, sketches)
    inferred = inference_occupancy(P_from(start), toy.reward, toy.gamma, toy.beta)
    assert np.allclose(info["y"], trained, atol=1e-10)
    assert np.linalg.norm(info["y"] - inferred) > 1e-2


def P_from(logits):
    return transition_kernel(logits)


def test_soft_aggregation_matches_equation_8():
    toy, _, P, _ = _toy_solution()
    identity = np.eye(toy.n_states)
    reward_hat, P_hat, gamma_hat, mu = aggregate_mdp(P, toy.reward, toy.gamma, identity)
    assert np.allclose(mu, np.ones(toy.n_states))
    assert np.allclose(reward_hat, toy.reward)
    assert np.allclose(P_hat, P)
    assert np.allclose(gamma_hat, toy.gamma)
    membership = np.asarray(softmax_membership(np.array([[0.2, -0.4], [1.0, 0.3], [-0.2, 0.7]])))
    assert np.allclose(membership.sum(axis=1), 1.0)
    reward_hat, P_hat, gamma_hat, mu = aggregate_mdp(P, toy.reward, toy.gamma, membership)
    assert np.allclose(mu, membership.sum(axis=0))
    for cluster in range(membership.shape[1]):
        for action in range(toy.n_actions):
            expected_reward = np.sum(membership[:, cluster] * toy.reward[:, action]) / mu[cluster]
            assert reward_hat[cluster, action] == pytest.approx(expected_reward)
            for nxt in range(membership.shape[1]):
                total = 0.0
                for state in range(toy.n_states):
                    for successor in range(toy.n_states):
                        total += (
                            membership[state, cluster]
                            * P[state, action, successor]
                            * membership[successor, nxt]
                        )
                assert P_hat[cluster, action, nxt] == pytest.approx(total / mu[cluster])
    assert np.allclose(P_hat.sum(axis=-1), 1.0)
    assert gamma_hat == pytest.approx(membership.T @ toy.gamma)
    y_hat = np.array([[0.4, 0.1], [0.2, 0.3]])
    lifted = lift_occupancy(y_hat, membership, mu)
    weights = y_hat / mu[:, None]
    assert np.allclose(lifted, membership @ weights)
