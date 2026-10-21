# Jacobian lens: replication on GPT-2

A from-scratch implementation of the **Jacobian lens** (Anthropic, 2026, *Verbalizable Representations Form a
Global Workspace in Language Models*) on open-weight GPT-2, built as a BlueDot Technical AI Safety project.

The lens decodes an intermediate residual stream `h_l` as `softmax(W_U · ln_f(J_l h_l))`, where `J_l` is the
Jacobian of the final residual stream with respect to `h_l`, **computed** by backpropagation and averaged over
prompts. Nothing is trained. With `J_l = I` it reduces exactly to the logit lens.

## What's here

| File | |
|---|---|
| `jlens_gpt2.ipynb` | Colab notebook (T4): correctness checks on gpt2-small, then the 100-prompt run on gpt2-medium, convergence curves, and a J-lens vs logit-lens comparison on held-out prompts |
| `build_nb.py` | Generates the notebook |
| `fix_paper_estimator.py` | The paper's all-positions estimator (`J_l = mean_t Σ_{t'≥t} ∂h_L[t']/∂h_l[t]`), same cost; the notebook's first run uses the last-token, same-position estimator |

## Correctness checks (gpt2-small, float64, one prompt, before any averaging)

- Final-layer logit lens reproduces the model's logits exactly (max diff 0.0).
- `J_l` vs central finite differences: relative error ~1e-7 at layers 0, 4, 8, 11.
- `J_L = I` exactly; `J_{L-1}` matches the last block's own Jacobian to 1e-15.
- Causal mask: backprop from position `t` gives exactly zero gradient at every position after `t`.
- All-positions estimator matches a brute-force per-position computation.

## Cost

All layers' Jacobians come out of the same `d_model` backward passes (batched 64 at a time), so the cost is set
by the earliest layer, not the number of layers. Measured: **~25 s per 64-token prompt for gpt2-medium (24
layers, d_model 1024) on a free Colab T4**, all 25 layer Jacobians at once.

## Status

October 2026: first code and first numbers. Next: results of the 100-prompt run, the all-positions estimator,
then scaling to a larger open-weight model.
