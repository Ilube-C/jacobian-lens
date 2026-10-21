# Paper's estimator: J_l = mean_t sum_{t'>=t} d h_L[t'] / d h_l[t]  (prompts averaged outside).
# One backward per output coordinate k of sum_{t'} h_L[t', k]; the gradient at source position t is then
# sum_{t'>=t} d h_L[t', k] / d h_l[t] (later t' only, by the causal mask). Average over t.
# Normalisation: with this choice J_L = I, so the logit lens is still J = I. ln_f is scale-invariant anyway.
def jacobians_one_prompt(model, ids, B=64, layers=None):
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
        out = hs[-1][torch.arange(b), :, ks].sum()
        out.backward()
        for j, l in enumerate(layers):
            if l == L:
                J[j, k0:k0 + b] = torch.eye(d, device=ids.device, dtype=J.dtype)[ks]
            else:
                g = hs[l - start].grad if l - start > 0 else h0.grad
                J[j, k0:k0 + b] = g.mean(dim=1)
    return J

MODEL_TAG = 'gpt2-medium-allpos'
SNAP_LAYERS = list(range(0, Lm + 1, 2))
SNAP_AT = set(list(range(1, 11)) + list(range(15, N_FIT + 1, 5)))
ck = f'{SAVE_DIR}/{MODEL_TAG}_ckpt.pt'
if os.path.exists(ck):
    state = torch.load(ck); print('resuming from prompt', state['n'])
else:
    state = dict(n=0, sumA=torch.zeros(Lm + 1, dm, dm), sumB=torch.zeros(Lm + 1, dm, dm), secs=[])
for n in range(state['n'], N_FIT):
    t0 = time.time()
    Jp = jacobians_one_prompt(med, fit_prompts[n].to(dev), B=64).cpu()
    (state['sumA'] if n % 2 == 0 else state['sumB']).add_(Jp)
    state['n'] = n + 1
    state['secs'].append(time.time() - t0)
    if n + 1 in SNAP_AT:
        torch.save(((state['sumA'] + state['sumB'])[SNAP_LAYERS] / (n + 1)).half(), f'{SAVE_DIR}/{MODEL_TAG}_snap{n + 1:03d}.pt')
    torch.save(state, ck + '.tmp'); os.replace(ck + '.tmp', ck)
    print(f'prompt {n + 1}/{N_FIT}  {state["secs"][-1]:.1f}s  mean {np.mean(state["secs"]):.1f}s/prompt', flush=True)
MODEL = MODEL_TAG   # so the results cells read these snapshots and name outputs accordingly
