"""H200 smoke for one occupancy-LP training loop.

Thirty Adam steps on the toy MDP. The simplex solve is NumPy. JAX builds the
sketched constraint and the fixed-basis Jacobian, and the job must see a GPU.
Not a paper benchmark.
"""

import jax
import numpy as np

from dfl_occupancy.occupancy import solve_occupancy_lp, transition_kernel
from dfl_occupancy.sketch import gaussian_row_sketches
from dfl_occupancy.toy import make_toy_mdp, same_basis_perturbation
from dfl_occupancy.train import train_step


def main():
    devices = jax.devices()
    print("jax", jax.__version__, devices, flush=True)
    if not any(d.platform == "gpu" for d in devices):
        raise SystemExit("expected a GPU device")

    toy = make_toy_mdp(0)
    true = solve_occupancy_lp(
        transition_kernel(toy.logits), toy.reward, toy.gamma, toy.beta
    )
    logits = same_basis_perturbation(toy.logits)
    sketches = np.asarray(
        gaussian_row_sketches(jax.random.PRNGKey(0), 4, 1, toy.n_states)
    )
    state = None
    first = None
    last = None
    steps = 30
    for step in range(steps):
        logits, state, info = train_step(
            logits,
            toy.reward,
            toy.gamma,
            toy.beta,
            sketches,
            toy.reward,
            true.H,
            true.dual,
            rho=1.0,
            adam_state=state,
            lr=5e-3,
        )
        loss = float(info["loss"])
        if step == 0:
            first = loss
        last = loss
        if step == 0 or step == steps - 1 or step % 10 == 0:
            print(f"step {step} surrogate {loss:.8e}", flush=True)
    print(f"train steps {steps}", flush=True)
    print(f"initial surrogate {first:.8e}", flush=True)
    print(f"final surrogate {last:.8e}", flush=True)
    if first != first or last != last:
        raise SystemExit("non-finite surrogate")
    print("dfl-occupancy-lp gpu smoke done", flush=True)


if __name__ == "__main__":
    main()
