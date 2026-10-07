"""One tabular training step on the toy MDP.

Predicted transitions are a softmax of logits. The forward occupancy is the
average of sketched solutions (Equation 6). The parameter step is Adam on the
augmented Lagrangian (Equation 5), using the average of the fixed-basis
Jacobians. The paper states that average on regions where every selected basis
is stable. Rewards are treated as known. This is not a paper benchmark.
"""

from __future__ import annotations

import numpy as np

from dfl_occupancy.sketch import jax_sketched_matrix, solve_sketched_lp
from dfl_occupancy.surrogate import augmented_lagrangian, augmented_lagrangian_grad_y
from dfl_occupancy.occupancy import fixed_basis_loss_gradient, transition_kernel


def adam_init(shape):
    return {
        "m": np.zeros(shape, dtype=np.float64),
        "v": np.zeros(shape, dtype=np.float64),
        "t": 0,
    }


def adam_apply(param, grad, state, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    grad = np.asarray(grad, dtype=np.float64)
    t = int(state["t"]) + 1
    m = b1 * state["m"] + (1.0 - b1) * grad
    v = b2 * state["v"] + (1.0 - b2) * np.square(grad)
    m_hat = m / (1.0 - b1**t)
    v_hat = v / (1.0 - b2**t)
    updated = np.asarray(param, dtype=np.float64) - lr * m_hat / (np.sqrt(v_hat) + eps)
    return updated, {"m": m, "v": v, "t": t}


def train_step(
    logits,
    reward,
    gamma,
    beta,
    sketches,
    r_true,
    H_true,
    nu_true,
    rho=1.0,
    adam_state=None,
    lr=1e-2,
):
    """One Adam step. The occupancy used in the loss is the sketch average."""
    P = transition_kernel(logits)
    sketches = np.asarray(sketches, dtype=np.float64)
    solutions = [solve_sketched_lp(P, reward, gamma, beta, sketch) for sketch in sketches]
    y_bar = np.mean([solution.y_flat for solution in solutions], axis=0)
    loss = augmented_lagrangian(y_bar, r_true, H_true, gamma, nu_true, rho)
    grad_y = augmented_lagrangian_grad_y(y_bar, r_true, H_true, gamma, nu_true, rho)
    jacobian = None
    for solution, sketch in zip(solutions, sketches):

        def constraint(lg, sketch=sketch):
            return jax_sketched_matrix(lg, beta, sketch)

        piece = fixed_basis_loss_gradient(
            grad_y, constraint, logits, solution.y_flat, solution.basis
        )
        # fixed_basis_loss_gradient already contracts grad_y. Average those.
        jacobian = piece if jacobian is None else jacobian + piece
    grad = np.asarray(jacobian / len(solutions), dtype=np.float64)
    if adam_state is None:
        adam_state = adam_init(np.shape(logits))
    new_logits, adam_state = adam_apply(logits, grad, adam_state, lr=lr)
    info = {
        "loss": loss,
        "y": y_bar.reshape(np.shape(reward)),
        "grad": grad,
    }
    return new_logits, adam_state, info


def _main():
    import jax

    from dfl_occupancy.sketch import gaussian_row_sketches
    from dfl_occupancy.toy import make_toy_mdp, same_basis_perturbation
    from dfl_occupancy.occupancy import solve_occupancy_lp

    toy = make_toy_mdp(0)
    true = solve_occupancy_lp(
        transition_kernel(toy.logits), toy.reward, toy.gamma, toy.beta
    )
    logits = same_basis_perturbation(toy.logits)
    sketches = np.asarray(
        gaussian_row_sketches(jax.random.PRNGKey(0), 4, 1, toy.n_states)
    )
    state = None
    rho = 1.0
    first = None
    last = None
    for step in range(200):
        logits, state, info = train_step(
            logits,
            toy.reward,
            toy.gamma,
            toy.beta,
            sketches,
            toy.reward,
            true.H,
            true.dual,
            rho=rho,
            adam_state=state,
            lr=5e-3,
        )
        if step == 0:
            first = info["loss"]
        last = info["loss"]
    print(f"train steps 200")
    print(f"initial surrogate {first:.8e}")
    print(f"final surrogate {last:.8e}")


if __name__ == "__main__":
    _main()
