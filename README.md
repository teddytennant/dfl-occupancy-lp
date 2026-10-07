# dfl_occupancy

A small JAX package for the occupancy-measure LP layer in decision-focused MDP learning from [arXiv:2610.08384](https://arxiv.org/abs/2610.08384).

## Install

```bash
pip install -e ".[test]"
```

## Run

```bash
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python demo.py
```

## What does not match the paper

Formulas follow the v1 PDF. The Inventory, Cliff Walking, and CartPole tasks are not implemented, and this package does not report those experimental numbers. Continuous-state occupancy features and the sample-based integrals in Section 5.2 are omitted. Discrete soft aggregation from Equations 7 and 8 is included as a closed form, but it is not trained on the paper's datasets.

The forward solve is a two-phase revised simplex with Bland's rule, with lexicographic basis enumeration as a fallback on these tiny LPs. The paper uses a pivoting algorithm and asks for a deterministic tie-break, but it does not name this rule. The penalty rho defaults to 1.0. The paper treats rho as an input chosen by validation and does not state a universal value. The predictor in the optional train step is a tabular softmax over transitions, not the task networks in the paper.

The abstract points to https://github.com/A-Eshragh/State_Aggregation_Project. That repository returned HTTP 404, so this package does not wrap upstream code.

## GPU smoke

Job 757605 ran on compute-gpu-02, one H200, jax 0.11.1, `CudaDevice(id=0)`. `smoke_gpu.py` took 30 Adam steps on the toy MDP. The sketched surrogate went from 8.30069960 to 4.81614612. The simplex solve is NumPy. JAX builds the sketched constraint and the fixed-basis Jacobian, and that part saw the GPU. This is not the Inventory, Cliff Walking, or CartPole result.
