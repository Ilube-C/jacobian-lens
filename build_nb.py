import json
cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.strip()})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip()})

md("""
# Jacobian lens: gpt2-small checks, then gpt2-medium over 100 prompts

Runtime → Change runtime type → **T4 GPU**, then run all cells in order.

Convention: `J_l[out, in] = d h_L[out] / d h_l[in]` at the **last token position**, where `h_l` is the
residual stream after `l` blocks (`l = 0` is the embedding, `l = L` is the final stream, before `ln_f`).
`J_l` is *computed* by backprop and averaged over prompts. Nothing here is trained.

Lens decode: `logits = W_U · ln_f(J_l h_l)`. With `J_l = I` this is exactly the logit lens.
""")

code("""
import torch, time, math, os, json, random
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(dev, torch.cuda.get_device_name(0) if dev == 'cuda' else '')
SAVE_DIR = '/content/jlens'
os.makedirs(SAVE_DIR, exist_ok=True)
""")

md("""
## Shared plumbing
A manual forward over the blocks so every residual stream `h_0 .. h_L` is a tensor we hold.
(HF's `output_hidden_states` applies `ln_f` to the last entry, which is the wrong thing to differentiate.)
""")

code("""
def load(name, dtype=torch.float32):
    tok = GPT2TokenizerFast.from_pretrained(name)
    model = GPT2LMHeadModel.from_pretrained(name, torch_dtype=dtype).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model

def embed(model, ids):
    pos = torch.arange(ids.shape[1], device=ids.device)
    return model.transformer.wte(ids) + model.transformer.wpe(pos)

def block_fwd(block, h):
    out = block(h)
    return out[0] if isinstance(out, tuple) else out   # newer transformers return a tensor

def run_blocks(model, h, start=0):
    # returns [h_start, h_start+1, ..., h_L]
    hs = [h]
    for block in model.transformer.h[start:]:
        h = block_fwd(block, h)
        hs.append(h)
    return hs

def decode(model, h):
    # logits from a residual-stream vector: W_U . ln_f(h)
    return model.lm_head(model.transformer.ln_f(h))
""")

md("""
## Step 1: logit lens on gpt2-small
The final layer must reproduce the model's own logits exactly. Earlier layers show the usual logit-lens picture.
""")

code("""
tok, small = load('gpt2')
L = small.config.n_layer; d = small.config.n_embd
print('gpt2-small: layers', L, 'd_model', d)

prompt = 'The Eiffel Tower is located in the city of'
ids = tok(prompt, return_tensors='pt').input_ids.to(dev)
with torch.no_grad():
    hs = run_blocks(small, embed(small, ids))
    ref = small(ids).logits[0, -1]
    lens_final = decode(small, hs[-1])[0, -1]
print('len(hs) =', len(hs), '(expect L+1 =', L + 1, ')')
print('final-layer logit lens vs model logits, max abs diff:', (lens_final - ref).abs().max().item())
for l in range(L + 1):
    with torch.no_grad():
        top = decode(small, hs[l])[0, -1].topk(5).indices
    print(f'layer {l:2d}:', [tok.decode(t) for t in top])
""")

md("""
## Step 2: compute `J_l` for one prompt, all layers from the same backward passes

Trick: batch `B` copies of the prompt. Copy `b` back-propagates output coordinate `k_b` at the last position,
so one backward pass yields `B` rows of every layer's Jacobian. Total cost is `d_model` rows,
**not** `d_model` per layer.

**Shape predictions before running:** each `J_l` is `(d, d)`. The gradient w.r.t. `h_l` is `(B, T, d)`.
Rows at positions `< T-1` are generally **non-zero** (attention reads earlier positions); there are no
positions after the last token, so nothing is masked out here. If we instead backpropagated from position `t`
in the middle, every position `> t` would get exactly zero gradient (causal mask). That is checked below.
""")

code("""
def jacobians_one_prompt(model, ids, B=64, pos=-1, layers=None):
    # Returns tensor J of shape (n_layers_kept, d, d) with J[l][out, in] = d h_L[pos, out] / d h_l[pos, in]
    d = model.config.n_embd; L = model.config.n_layer
    layers = list(range(L + 1)) if layers is None else layers
    J = torch.zeros(len(layers), d, d, device=ids.device, dtype=model.transformer.wte.weight.dtype)
    start = min(layers)
    with torch.no_grad():
        h_start = run_blocks(model, embed(model, ids))[start] if start > 0 else embed(model, ids)
    for k0 in range(0, d, B):
        ks = torch.arange(k0, min(k0 + B, d), device=ids.device)
        b = len(ks)
        h0 = h_start.expand(b, -1, -1).clone().requires_grad_(True)
        hs = [h0]
        h = h0
        for block in model.transformer.h[start:]:
            h = block_fwd(block, h)
            h.retain_grad()
            hs.append(h)
        # hs[i] is h_{start+i}
        out = hs[-1][torch.arange(b), pos, ks].sum()
        out.backward()
        for j, l in enumerate(layers):
            g = hs[l - start].grad if l - start > 0 else h0.grad
            if l == L:
                g = torch.zeros_like(hs[-1]); g[torch.arange(b), pos, ks] = 1.0   # d h_L / d h_L
            J[j, k0:k0 + b] = g[:, pos]
    return J

t0 = time.time()
J1 = jacobians_one_prompt(small, ids)
print('J shape', tuple(J1.shape), f'{time.time() - t0:.1f}s')
""")

md("""
## Step 3: finite-difference check (the important one)
Perturb `h_l[last] += eps * e_i`, run the remaining blocks, compare `Δh_L[last] / eps` with column `J_l[:, i]`.
Done in float64 so eps can be small. One prompt, before any averaging.
""")

code("""
small64 = small.double()
with torch.no_grad():
    hs64 = run_blocks(small64, embed(small64, ids))
J64 = jacobians_one_prompt(small64, ids, layers=[0, 4, 8, 11, 12])
eps = 1e-6
rng = np.random.default_rng(0)
for j, l in enumerate([0, 4, 8, 11, 12]):
    errs = []
    for i in rng.choice(d, 5, replace=False):
        hp = hs64[l].clone(); hp[0, -1, i] += eps
        hm = hs64[l].clone(); hm[0, -1, i] -= eps
        with torch.no_grad():
            fd = (run_blocks(small64, hp, start=l)[-1][0, -1] - run_blocks(small64, hm, start=l)[-1][0, -1]) / (2 * eps)
        col = J64[j][:, i]
        errs.append(((fd - col).norm() / col.norm()).item())
    print(f'layer {l:2d}: relative FD error per column', ['%.1e' % e for e in errs])
""")

md("""
## Step 4: structural checks
* `J_L` must be exactly `I`.
* `J_{L-1}` must equal `I +` (Jacobian of the last block alone), computed independently through that one block.
* Causal mask: backprop from a middle position `t`; gradients at positions `> t` must be exactly zero.
""")

code("""
Lh = small64.config.n_layer
print('||J_L - I|| =', (J64[-1] - torch.eye(d, device=dev, dtype=torch.float64)).norm().item())

# independent single-block Jacobian at the last position
hL1 = hs64[Lh - 1].detach()
def last_block_out(x_last):
    x = hL1.clone(); x[0, -1] = x_last
    return block_fwd(small64.transformer.h[Lh - 1], x)[0, -1]
Jblock = torch.autograd.functional.jacobian(last_block_out, hL1[0, -1])
J_Lm1 = jacobians_one_prompt(small64, ids, layers=[Lh - 1, Lh])[0]
print('||J_{L-1} - Jblock|| / ||Jblock|| =', ((J_Lm1 - Jblock).norm() / Jblock.norm()).item())
print('(Jblock already includes the residual identity, so this should be ~1e-12)')
print('||J_{L-1} - I|| =', (J_Lm1 - torch.eye(d, device=dev, dtype=torch.float64)).norm().item())

# causal mask
t = ids.shape[1] // 2
h0 = embed(small64, ids).detach().requires_grad_(True)
run_blocks(small64, h0)[-1][0, t].sum().backward()
gnorm = h0.grad[0].norm(dim=-1)
print('grad norm per position (backprop from t=%d):' % t, ['%.2g' % x for x in gnorm.tolist()])
print('all positions > t exactly zero:', bool((gnorm[t + 1:] == 0).all()))
small = small64.float()
""")

md("""
## Step 5: the headline run on gpt2-medium
100 prompts for `J`, 20 held-out for evaluation. Checkpoints after every prompt, so a disconnect only loses
one prompt: rerun this cell and it resumes. Optionally mount Drive (next cell) so checkpoints survive the VM.
""")

code("""
# Optional: persist checkpoints to Drive. Skip this cell if the Colab account's Drive is not one you want to use.
from google.colab import drive
drive.mount('/content/drive')
SAVE_DIR = '/content/drive/MyDrive/jlens'
os.makedirs(SAVE_DIR, exist_ok=True)
""")

code("""
MODEL = 'gpt2-medium'; N_FIT = 100; N_EVAL = 20; SEQ = 64
tokm, med = load(MODEL)
Lm, dm = med.config.n_layer, med.config.n_embd
print(MODEL, 'layers', Lm, 'd_model', dm)

from datasets import load_dataset
ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='validation')
texts = [x['text'].strip() for x in ds if len(x['text'].split()) > 80 and not x['text'].strip().startswith('=')]
random.Random(0).shuffle(texts)
prompts = []
for tx in texts:
    ids_ = tokm(tx, return_tensors='pt').input_ids[:, :SEQ]
    if ids_.shape[1] == SEQ:
        prompts.append(ids_)
    if len(prompts) == N_FIT + N_EVAL:
        break
fit_prompts, eval_prompts = prompts[:N_FIT], prompts[N_FIT:]
print(len(fit_prompts), 'fit prompts,', len(eval_prompts), 'eval prompts, seq len', SEQ)
""")

code("""
SNAP_LAYERS = list(range(0, Lm + 1, 2))          # snapshots keep every other layer, to stay small
SNAP_AT = set(list(range(1, 11)) + list(range(15, N_FIT + 1, 5)))
ck = f'{SAVE_DIR}/{MODEL}_ckpt.pt'
if os.path.exists(ck):
    state = torch.load(ck)
    print('resuming from prompt', state['n'])
else:
    # sums over even- and odd-indexed prompts: their sum is the full sum, their difference the noise floor
    state = dict(n=0, sumA=torch.zeros(Lm + 1, dm, dm), sumB=torch.zeros(Lm + 1, dm, dm), secs=[])

eye = torch.eye(dm)
for n in range(state['n'], N_FIT):
    t0 = time.time()
    Jp = jacobians_one_prompt(med, fit_prompts[n].to(dev), B=64).cpu()
    (state['sumA'] if n % 2 == 0 else state['sumB']).add_(Jp)
    state['n'] = n + 1
    state['secs'].append(time.time() - t0)
    if n + 1 in SNAP_AT:
        torch.save(((state['sumA'] + state['sumB'])[SNAP_LAYERS] / (n + 1)).half(), f'{SAVE_DIR}/{MODEL}_snap{n + 1:03d}.pt')
    torch.save(state, ck + '.tmp'); os.replace(ck + '.tmp', ck)
    print(f'prompt {n + 1}/{N_FIT}  {state["secs"][-1]:.1f}s  mean {np.mean(state["secs"]):.1f}s/prompt', flush=True)
""")

md("""
## Step 6: results
* convergence curve per layer: `||J_n - J_N||_F / ||J_N - I||_F` vs `n` (relative error of the correction)
* cross-half floor: `||J_A - J_B||_F / ||J_N - I||_F` per layer
* `||J_l - I||_F` across depth (should shrink towards 0 at the top)
* timing: seconds per prompt
""")

code("""
N = state['n']; Jfin = (state['sumA'] + state['sumB']) / N
running = [(n, torch.load(f'{SAVE_DIR}/{MODEL}_snap{n:03d}.pt')) for n in sorted(SNAP_AT) if n <= N]
corr = torch.stack([(Jfin[l] - eye).norm() for l in range(Lm + 1)])
JA = state['sumA'] / max(1, (N + 1) // 2); JB = state['sumB'] / max(1, N // 2)
floor = torch.stack([(JA[l] - JB[l]).norm() for l in range(Lm + 1)]) / corr.clamp_min(1e-12)

layers_show = [0, 4, 8, 12, 16, 20, 22]
ns = [n for n, _ in running]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
for l in layers_show:
    si = SNAP_LAYERS.index(l)
    errs = [((Jn[si].float() - Jfin[l]).norm() / corr[l]).item() for _, Jn in running]
    ax[0].loglog(ns[:-1], errs[:-1], marker='o', ms=3, label=f'layer {l}')
ref = np.array(ns[:-1], dtype=float)
ax[0].loglog(ref, 0.5 * np.sqrt(ref[0] / ref) , 'k--', lw=1, label='1/sqrt(n)')
ax[0].set_xlabel('prompts n'); ax[0].set_ylabel('||J_n - J_N|| / ||J_N - I||'); ax[0].set_title('Convergence per layer'); ax[0].legend(fontsize=7)
ax[1].plot(range(Lm + 1), corr.numpy(), marker='o'); ax[1].set_xlabel('layer l'); ax[1].set_ylabel('||J_l - I||_F'); ax[1].set_title('Size of the correction')
ax[2].plot(range(Lm), floor[:Lm].numpy(), marker='o'); ax[2].set_xlabel('layer l'); ax[2].set_ylabel('||J_A - J_B|| / ||J - I||'); ax[2].set_title('Cross-half noise floor')
plt.tight_layout(); plt.savefig(f'{SAVE_DIR}/{MODEL}_results.png', dpi=130); plt.show()

half = [J for n, J in running if n == N // 2]   # error of the half-corpus estimate vs the full one
e4 = ((half[0][SNAP_LAYERS.index(4)].float() - Jfin[4]).norm() / corr[4]).item() if half else float('nan')
e20 = ((half[0][SNAP_LAYERS.index(20)].float() - Jfin[20]).norm() / corr[20]).item() if half else float('nan')
print(f'prompts: {N}   mean s/prompt: {np.mean(state["secs"]):.1f}   total min: {sum(state["secs"]) / 60:.1f}')
print(f'cross-half floor, layer 4: {floor[4]:.3f}   layer 20: {floor[20]:.3f}   ratio: {floor[4] / floor[20]:.2f}  (paper target ~4x)')
print(f'error at n={N // 2} vs n={N}, layer 4: {e4:.3f}  layer 20: {e20:.3f}  ratio: {e4 / e20:.2f}')
""")

md("""
## Step 7: does the J-lens beat the logit lens? (held-out prompts)
For each layer: KL(final model distribution || lens distribution) and top-1 agreement, at every position of the
20 held-out prompts. The J-lens should be at or below the logit lens everywhere, by most at early layers.
""")

code("""
Jdev = Jfin.to(dev)
kl_logit = torch.zeros(Lm + 1); kl_j = torch.zeros(Lm + 1); top_logit = torch.zeros(Lm + 1); top_j = torch.zeros(Lm + 1)
cnt = 0
with torch.no_grad():
    for p in eval_prompts:
        hs = run_blocks(med, embed(med, p.to(dev)))
        lp_final = torch.log_softmax(decode(med, hs[-1])[0], -1)
        tgt = lp_final.argmax(-1)
        for l in range(Lm + 1):
            for name, h in (('logit', hs[l][0]), ('j', hs[l][0] @ Jdev[l].T)):
                lp = torch.log_softmax(decode(med, h), -1)
                kl = (lp_final.exp() * (lp_final - lp)).sum(-1).mean().item()
                top = (lp.argmax(-1) == tgt).float().mean().item()
                if name == 'logit': kl_logit[l] += kl; top_logit[l] += top
                else: kl_j[l] += kl; top_j[l] += top
        cnt += 1
kl_logit /= cnt; kl_j /= cnt; top_logit /= cnt; top_j /= cnt
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(kl_logit, marker='o', label='logit lens'); ax[0].plot(kl_j, marker='o', label='J-lens'); ax[0].set_yscale('log'); ax[0].set_xlabel('layer'); ax[0].set_ylabel('KL(final || lens)'); ax[0].legend()
ax[1].plot(top_logit, marker='o', label='logit lens'); ax[1].plot(top_j, marker='o', label='J-lens'); ax[1].set_xlabel('layer'); ax[1].set_ylabel('top-1 agreement with final'); ax[1].legend()
plt.tight_layout(); plt.savefig(f'{SAVE_DIR}/{MODEL}_lens_vs_logit.png', dpi=130); plt.show()
json.dump(dict(model=MODEL, n=N, secs_per_prompt=float(np.mean(state['secs'])), corr=corr.tolist(), floor=floor.tolist(),
               kl_logit=kl_logit.tolist(), kl_j=kl_j.tolist(), top_logit=top_logit.tolist(), top_j=top_j.tolist()),
          open(f'{SAVE_DIR}/{MODEL}_summary.json', 'w'), indent=1)
""")

nb = {"cells":cells,"metadata":{"accelerator":"GPU","colab":{"gpuType":"T4","provenance":[]},"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":0}
json.dump(nb, open("jlens_gpt2.ipynb","w"), indent=1)
print("ok", len(cells))
