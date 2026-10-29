"""Round 36: record OLMo-2 MLP neurons over a fixed document pool at one checkpoint (PREDICTIONS.md, round 36).

pool: tokenize the first D documents of a parquet `text` column once per model (the base model's tokenizer), so every
checkpoint sees the same ids. Document k's ids are tokenizer(text, add_special_tokens=True)['input_ids'][:256]; a
position counts (is valid) iff it is >= 1 and its id is not in tokenizer.all_special_ids. Documents with fewer than 2
ids are dropped before numbering. Writes POOL.npz: ids int32 (all documents concatenated), off int64 [D+1] (document k
is ids[off[k]:off[k+1]]), valid bool (same length as ids).

units(repo, n_layers, width): per layer l, 32 neurons
    np.random.default_rng(zlib.crc32(f'r36:{repo}:L{l}'.encode())).choice(width, 32, replace=False)
then 8 directions: the columns of default_rng(crc32(f'r36:dir:{repo}:L{l}')).standard_normal((width, 8)), each
scaled to unit norm. Unit order: layer 0's 32 neurons (names L0:n<i>), layer 0's 8 directions (L0:d0..d7), layer 1...
repo is always the base repo (post-training checkpoints use the base model's units).

record: one checkpoint over the pool. The activation of neuron (l, i) on a token is entry i of the input of
model.model.layers[l].mlp.down_proj; a direction's is the dot product of that input (as float32) with its vector.
Documents run in pool order, B at a time, right-padded, bf16 weights, no gradients. Writes OUT.npz:
    top_val, top_doc, top_pos  [L, 32, W]  for every neuron, its 32 documents with the highest per-document max
                                            (max over valid positions), descending, with that max's position
    bot_val, bot_doc, bot_pos  [L, 32, W]  the same for the lowest per-document min, ascending
    tok_mean, tok_std          [L, W]      over all valid tokens of the pool (population std)
    u_max, u_argmax            [D, U]      every drawn unit's per-document max over valid positions, and its position
    u_min, u_argmin            [D, U]      the same for the min (first position on ties, as torch.max/min do)
    u_tok                      [N, U]      every drawn unit's value on every position of documents 0..min(D,1000)-1
                                            (N = off[min(D,1000)]); NaN at invalid positions
    units                      [U]         unit names; meta: json string (model directory, units_repo, n_docs)
Values float32 except top/bot (float16 values, int16 document and position) and u_tok (float16).

reps: the main checkpoint's MLP input (the residual stream after attention: the input of layers[l].mlp) at given
(layer, document, position) triples (TRIPLES.npy, int64 [n, 3]), plus every layer's per-dimension mean and std of that
input over the valid tokens of documents 0-199. Writes OUT.npz: reps float16 [n, d], mean, std float32 [L, d].

usage: python -I tools/r36_record.py pool --tokenizer DIR --parquet P --out POOL.npz [--docs 10000]
       python -I tools/r36_record.py record --model DIR --units-repo REPO --pool POOL.npz --out OUT.npz [--batch 16]
       python -I tools/r36_record.py reps --model DIR --pool POOL.npz --triples T.npy --out OUT.npz [--batch 16]
"""
import argparse
import json
import zlib

import numpy as np

MAXTOK, K, NN, ND, TOKDOCS = 256, 32, 32, 8, 1000


def units(repo, n_layers, width):
    """[(layer, neuron indices [32], directions [width, 8])], names"""
    out, names = [], []
    for l in range(n_layers):
        idx = np.random.default_rng(zlib.crc32(f'r36:{repo}:L{l}'.encode())).choice(width, NN, replace=False)
        w = np.random.default_rng(zlib.crc32(f'r36:dir:{repo}:L{l}'.encode())).standard_normal((width, ND))
        out.append((l, idx, w / np.linalg.norm(w, axis=0, keepdims=True)))
        names += [f'L{l}:n{int(i)}' for i in idx] + [f'L{l}:d{j}' for j in range(ND)]
    return out, names


def pool(tokenizer_dir, parquet, out, docs):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    special = set(tok.all_special_ids)
    ids, off, valid = [], [0], []
    for text in pq.read_table(parquet, columns=['text']).column('text').to_pylist():
        if len(off) > docs:
            break
        x = tok(text, add_special_tokens=True)['input_ids'][:MAXTOK]
        if len(x) < 2:
            continue
        ids += x
        valid += [j >= 1 and t not in special for j, t in enumerate(x)]
        off.append(len(ids))
    np.savez(out, ids=np.array(ids, np.int32), off=np.array(off, np.int64), valid=np.array(valid, bool))
    print('docs', len(off) - 1, 'tokens', len(ids), 'valid', int(np.sum(valid)))


def load(model_dir, device):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    return m.to(device).eval()


def batches(P, B, device):
    """(first document, ids [b, T], valid [b, T], attention mask [b, T]) for consecutive documents, right-padded"""
    import torch
    ids, off, valid = P['ids'], P['off'], P['valid']
    D = len(off) - 1
    for a in range(0, D, B):
        ks = range(a, min(a + B, D))
        T = max(off[k + 1] - off[k] for k in ks)
        x = np.zeros((len(ks), T), np.int64)
        v = np.zeros((len(ks), T), bool)
        att = np.zeros((len(ks), T), np.int64)
        for r, k in enumerate(ks):
            n = off[k + 1] - off[k]
            x[r, :n], v[r, :n], att[r, :n] = ids[off[k]:off[k + 1]], valid[off[k]:off[k + 1]], 1
        yield a, torch.from_numpy(x).to(device), torch.from_numpy(v).to(device), torch.from_numpy(att).to(device)


def record(model_dir, units_repo, pool_path, out, B, device):
    import torch
    P = dict(np.load(pool_path))
    D, off = len(P['off']) - 1, P['off']
    m = load(model_dir, device)
    layers = m.model.layers
    L, W = len(layers), layers[0].mlp.down_proj.in_features
    U, names = units(units_repo, L, W)
    nU = NN + ND
    td = min(D, TOKDOCS)
    tv = torch.full((L, K, W), -torch.inf, device=device)
    tdoc = torch.zeros((L, K, W), dtype=torch.int64, device=device)
    tpos = torch.zeros_like(tdoc)
    bv = torch.full((L, K, W), torch.inf, device=device)
    bdoc, bpos = torch.zeros_like(tdoc), torch.zeros_like(tdoc)
    s1 = torch.zeros((L, W), dtype=torch.float64, device=device)
    s2 = torch.zeros_like(s1)
    cnt = [0]
    u_max = np.zeros((D, L * nU), np.float32)
    u_argmax = np.zeros((D, L * nU), np.int16)
    u_min, u_argmin = np.zeros_like(u_max), np.zeros_like(u_argmax)
    u_tok = np.full((off[td], L * nU), np.nan, np.float16)
    idx = [torch.from_numpy(i).to(device) for _, i, _ in U]
    dirs = [torch.from_numpy(w).float().to(device) for _, _, w in U]
    cur = {}

    def hook(l):
        def f(module, inputs):
            h = inputs[0]  # [b, T, W]
            a, v = cur['a'], cur['v']
            b = h.shape[0]
            docs = torch.arange(a, a + b, device=device)
            # all neurons: per-document max and min over valid positions, merged into the running top/bottom 32
            vm = v[..., None]
            mx, amx = h.masked_fill(~vm, -torch.inf).float().max(1)
            mn, amn = h.masked_fill(~vm, torch.inf).float().min(1)
            for val, doc, pos, x, ax, desc in ((tv, tdoc, tpos, mx, amx, True), (bv, bdoc, bpos, mn, amn, False)):
                cv = torch.cat([val[l], x])
                cd = torch.cat([doc[l], docs[:, None].expand(b, W)])
                cp = torch.cat([pos[l], ax])
                s, j = cv.topk(K, dim=0, largest=desc)
                val[l], doc[l], pos[l] = s, cd.gather(0, j), cp.gather(0, j)
            hf = h.float() * vm
            s1[l] += hf.sum((0, 1)).double()
            s2[l] += (hf * hf).sum((0, 1)).double()
            if l == 0:
                cnt[0] += int(v.sum())
            # drawn units
            uv = torch.cat([h[..., idx[l]].float(), h.float() @ dirs[l]], -1)  # [b, T, 40]
            umx, uamx = uv.masked_fill(~vm, -torch.inf).max(1)
            umn, uamn = uv.masked_fill(~vm, torch.inf).min(1)
            cols = slice(l * nU, (l + 1) * nU)
            u_max[a:a + b, cols], u_argmax[a:a + b, cols] = umx.cpu().numpy(), uamx.cpu().numpy()
            u_min[a:a + b, cols], u_argmin[a:a + b, cols] = umn.cpu().numpy(), uamn.cpu().numpy()
            if a < td:
                un = uv.masked_fill(~vm, torch.nan).half().cpu().numpy()
                for r in range(min(b, td - a)):
                    k = a + r
                    u_tok[off[k]:off[k + 1], cols] = un[r, :off[k + 1] - off[k]]
        return f

    hs = [layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in range(L)]
    with torch.inference_mode():
        for a, x, v, att in batches(P, B, device):
            cur['a'], cur['v'] = a, v
            m(input_ids=x, attention_mask=att, use_cache=False)
    for h in hs:
        h.remove()
    mean = (s1 / cnt[0])
    std = (s2 / cnt[0] - mean * mean).clamp_min(0).sqrt()
    np.savez(out, top_val=tv.half().cpu().numpy(), top_doc=tdoc.short().cpu().numpy(),
             top_pos=tpos.short().cpu().numpy(), bot_val=bv.half().cpu().numpy(), bot_doc=bdoc.short().cpu().numpy(),
             bot_pos=bpos.short().cpu().numpy(), tok_mean=mean.float().cpu().numpy(),
             tok_std=std.float().cpu().numpy(), u_max=u_max, u_argmax=u_argmax, u_min=u_min, u_argmin=u_argmin,
             u_tok=u_tok, units=np.array(names),
             meta=json.dumps({'model': str(model_dir), 'units_repo': units_repo, 'n_docs': D}))


def reps(model_dir, pool_path, triples, out, B, device):
    import torch
    P = dict(np.load(pool_path))
    t = np.load(triples)
    m = load(model_dir, device)
    layers = m.model.layers
    L, d = len(layers), m.config.hidden_size
    R = np.zeros((len(t), d), np.float16)
    s1 = torch.zeros((L, d), dtype=torch.float64, device=device)
    s2, cnt, cur = torch.zeros_like(s1), [0], {}
    need = {}
    for n, (l, k, p) in enumerate(t):
        need.setdefault(int(k), []).append((n, int(l), int(p)))

    def hook(l):
        def f(module, inputs):
            x, a, v = inputs[0], cur['a'], cur['v']
            if a < 200:
                hv = x.float() * v[..., None]
                s1[l] += hv.sum((0, 1)).double()
                s2[l] += (hv * hv).sum((0, 1)).double()
                if l == 0:
                    cnt[0] += int(v.sum())
            for r in range(x.shape[0]):
                for n, ll, p in need.get(a + r, ()):
                    if ll == l:
                        R[n] = x[r, p].float().cpu().numpy()
        return f

    hs = [layers[l].mlp.register_forward_pre_hook(hook(l)) for l in range(L)]
    docs = sorted(set(need) | set(range(min(200, len(P['off']) - 1))))
    with torch.inference_mode():
        for k in docs:  # one document at a time: positions are the pool's
            cur['a'] = k
            x = torch.from_numpy(P['ids'][P['off'][k]:P['off'][k + 1]].astype(np.int64))[None].to(device)
            cur['v'] = torch.from_numpy(P['valid'][P['off'][k]:P['off'][k + 1]])[None].to(device)
            m(input_ids=x, use_cache=False)
    for h in hs:
        h.remove()
    mean = s1 / cnt[0]
    std = (s2 / cnt[0] - mean * mean).clamp_min(0).sqrt()
    np.savez(out, reps=R, mean=mean.float().cpu().numpy(), std=std.float().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('pool', 'record', 'reps'))
    ap.add_argument('--tokenizer')
    ap.add_argument('--parquet')
    ap.add_argument('--docs', type=int, default=10000)
    ap.add_argument('--model')
    ap.add_argument('--units-repo')
    ap.add_argument('--pool')
    ap.add_argument('--triples')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    if a.cmd == 'pool':
        pool(a.tokenizer, a.parquet, a.out, a.docs)
    elif a.cmd == 'record':
        record(a.model, a.units_repo, a.pool, a.out, a.batch, a.device)
    else:
        reps(a.model, a.pool, a.triples, a.out, a.batch, a.device)


if __name__ == '__main__':
    main()
