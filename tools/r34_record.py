"""Record OLMo-2's internal state at one readout token, for a list of sentences.

Each sentence is run once (no padding, bf16 weights, no gradients, eager attention) on
    tokenizer(text, add_special_tokens=True) ids + tokenizer(suffix, add_special_tokens=False) ids
and at two readout positions r, "suffix" (the last suffix token, len(ids) - 1) and "text" (the sentence's own last
token, n_text - 1), it records for every layer l = 0..L-1:
    pre_<r>     [n, L, d]      residual stream into layers[l] (pre[0] is the token embedding)
    mid_<r>     [n, L, d]      residual stream between attention and MLP (the input of layers[l].mlp)
    post_<r>    [n, L, d]      residual stream out of layers[l]
    hidden_<r>  [n, L, 11008]  MLP hidden activation (the input of layers[l].mlp.down_proj)
    attn_<r>    per sentence [L, heads, len(ids)], attention weights from position r (zeros past r)
plus ids (per sentence) and n_text. decompose(pre, mid, post) gives each layer's attention and MLP writes.

Usage:
    python tools/r34_record.py --model DIR --stimuli FILE --out DIR [--suffix '\\n'] [--device cuda]
    python tools/r34_record.py --selftest

The stimuli file is a JSON list of records with at least `sid` and `text`; they are recorded in file order. Writes
out/record.npz (load with allow_pickle=True: attn_<r> and ids are object arrays) and out/sentences.json (the records
as given, plus each one's token strings). --suffix takes backslash escapes, so --suffix '\\n' is a newline.
"""
import argparse
import json
import os

import numpy as np

KINDS = ('pre', 'mid', 'post', 'hidden')
READOUTS = ('suffix', 'text')


def build_ids(tokenizer, text, suffix='\n'):
    """(ids, n_text): the text's ids with special tokens, then the suffix's ids without them"""
    head = [int(i) for i in tokenizer(text, add_special_tokens=True)['input_ids']]
    tail = [int(i) for i in tokenizer(suffix, add_special_tokens=False)['input_ids']]
    assert head, f'no ids for text {text!r}'
    assert tail, f'no ids for suffix {suffix!r}'
    return head + tail, len(head)


def readout_positions(ids, n_text):
    """{"suffix": the last suffix token, "text": the sentence's own last token}"""
    assert 1 <= n_text < len(ids), (n_text, len(ids))
    return {'suffix': len(ids) - 1, 'text': n_text - 1}


def decompose(pre, mid, post):
    """(attn_write, mlp_write) = (mid - pre, post - mid)"""
    pre, mid, post = np.asarray(pre), np.asarray(mid), np.asarray(post)
    assert pre.shape == mid.shape == post.shape, (pre.shape, mid.shape, post.shape)
    return mid - pre, post - mid


def objarr(items):
    """a 1-d object array holding each item as is (equal-length items are not stacked)"""
    a = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        a[i] = x
    return a


def unescape(s):
    """backslash escapes in a command-line string ('\\n' -> newline); other characters pass through"""
    return s.encode('latin-1', 'backslashreplace').decode('unicode_escape')


def add_hooks(layers, cache):
    """pre / mid / hidden from forward pre-hooks, post from a forward hook, into cache[(kind, l)]; returns handles"""
    def grab_in(kind, l, key):
        def fn(module, args, kwargs):
            x = args[0] if args else kwargs[key]
            cache[(kind, l)] = x.detach()
        return fn

    def grab_out(l):
        def fn(module, args, output):
            x = output[0] if isinstance(output, tuple) else output
            cache[('post', l)] = x.detach()
        return fn

    handles = []
    for l, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(grab_in('pre', l, 'hidden_states'), with_kwargs=True))
        handles.append(layer.mlp.register_forward_pre_hook(grab_in('mid', l, 'x'), with_kwargs=True))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(grab_in('hidden', l, 'input'),
                                                                     with_kwargs=True))
        handles.append(layer.register_forward_hook(grab_out(l)))
    return handles


def max_rel_err(a, b):
    """max over positions of ||a - b|| / ||b|| (norms over the last axis)"""
    a, b = a.float(), b.float()
    assert a.shape == b.shape, (tuple(a.shape), tuple(b.shape))
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)).max())


def check_hidden_states(cache, hidden_states, L, tol=1e-2):
    """post[l] == hidden state l + 1 for l < L - 1, and pre[0] == hidden state 0, at every position"""
    assert hidden_states is not None and len(hidden_states) == L + 1, 'expected L + 1 hidden states'
    err = max_rel_err(cache[('pre', 0)], hidden_states[0])
    assert err <= tol, f'pre[0] vs hidden state 0: relative error {err:.3g}'
    for l in range(L - 1):
        err = max_rel_err(cache[('post', l)], hidden_states[l + 1])
        assert err <= tol, f'post[{l}] vs hidden state {l + 1}: relative error {err:.3g}'


def record(model, tokenizer, sentences, suffix='\n', device='cuda'):
    """one forward pass per sentence; dict of numpy arrays (see the module docstring), plus per-sentence tokens"""
    import torch
    layers, cfg = model.model.layers, model.config
    L = len(layers)
    assert L == cfg.num_hidden_layers, (L, cfg.num_hidden_layers)
    assert getattr(cfg, '_attn_implementation', None) == 'eager', 'load the model with attn_implementation="eager"'
    assert next(model.parameters()).dtype == torch.bfloat16, 'load the model with bf16 weights'
    assert len(sentences) > 0
    width = {'pre': cfg.hidden_size, 'mid': cfg.hidden_size, 'post': cfg.hidden_size,
             'hidden': cfg.intermediate_size}
    acc = {f'{k}_{r}': [] for k in KINDS + ('attn',) for r in READOUTS}
    all_ids, n_texts, tokens = [], [], []
    cache = {}
    handles = add_hooks(layers, cache)
    try:
        with torch.no_grad():
            for i, s in enumerate(sentences):
                text = s['text'] if isinstance(s, dict) else s
                ids, n_text = build_ids(tokenizer, text, suffix)
                pos, T = readout_positions(ids, n_text), len(ids)
                cache.clear()
                out = model(input_ids=torch.tensor([ids], device=device), output_attentions=True,
                            output_hidden_states=(i == 0), use_cache=False)
                assert set(cache) == {(k, l) for k in KINDS for l in range(L)}, 'a hook did not fire'
                for (k, l), x in cache.items():
                    assert tuple(x.shape) == (1, T, width[k]), (k, l, tuple(x.shape))
                att = out.attentions
                assert att is not None and len(att) == L, 'no attention weights (needs eager attention)'
                for w in att:
                    assert w.shape[0] == 1 and w.shape[-2:] == (T, T), tuple(w.shape)
                if i == 0:
                    check_hidden_states(cache, out.hidden_states, L)
                for r, p in pos.items():
                    for k in KINDS:
                        x = torch.stack([cache[(k, l)][0, p] for l in range(L)])
                        acc[f'{k}_{r}'].append(x.float().cpu().numpy())
                    a = torch.stack([w[0, :, p, :] for w in att]).float().cpu().numpy()
                    a[:, :, p + 1:] = 0.0
                    acc[f'attn_{r}'].append(a)
                all_ids.append(np.array(ids, dtype=np.int64))
                n_texts.append(n_text)
                tokens.append(list(tokenizer.convert_ids_to_tokens(ids)))
                if (i + 1) % 50 == 0 or i + 1 == len(sentences):
                    print(f'recorded {i + 1}/{len(sentences)}', flush=True)
    finally:
        for h in handles:
            h.remove()
    rec = {k: objarr(v) if k.startswith('attn_') else np.stack(v).astype(np.float32) for k, v in acc.items()}
    rec['ids'] = objarr(all_ids)
    rec['n_text'] = np.array(n_texts, dtype=np.int64)
    rec['tokens'] = objarr(tokens)
    return rec


def save(out, rec, sentences):
    """out/record.npz (every array but tokens) and out/sentences.json (records as given + their token strings)"""
    assert len(sentences) == len(rec['n_text']) == len(rec['tokens'])
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, 'record.npz'), **{k: v for k, v in rec.items() if k != 'tokens'})
    recs = []
    for s, toks in zip(sentences, rec['tokens']):
        s = dict(s) if isinstance(s, dict) else {'text': s}
        assert 'tokens' not in s, 'a stimulus record already has a "tokens" field'
        recs.append({**s, 'tokens': list(toks)})
    with open(os.path.join(out, 'sentences.json'), 'w') as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)


def selftest():
    def fake_tok(text, add_special_tokens=True):
        return {'input_ids': ([1] if add_special_tokens else []) + [ord(c) for c in text]}

    ids, n = build_ids(fake_tok, 'ab')
    assert ids == [1, 97, 98, 10] and n == 3, (ids, n)
    assert readout_positions(ids, n) == {'suffix': 3, 'text': 2}
    ids2, n2 = build_ids(fake_tok, 'xyz', suffix=' A:')
    assert ids2 == [1, 120, 121, 122, 32, 65, 58] and n2 == 4, (ids2, n2)
    ids3, n3 = build_ids(fake_tok, 'q', suffix=' A:')
    assert ids2[n2:] == ids3[n3:] == [32, 65, 58]  # the suffix ids are the same for every sentence
    pos = readout_positions(ids2, n2)
    assert ids2[pos['suffix']] == 58 and ids2[pos['text']] == 122

    pre = np.arange(24.0).reshape(2, 3, 4)
    mid = pre + 1.0
    post = mid * 2.0
    aw, mw = decompose(pre, mid, post)
    assert aw.shape == mw.shape == pre.shape
    assert np.allclose(aw, 1.0) and np.allclose(mw, mid) and np.allclose(pre + aw + mw, post)

    assert unescape('\\n') == '\n' and unescape('\n') == '\n' and unescape('é中') == 'é中'
    a = objarr([np.zeros(2), np.ones(2)])
    assert a.shape == (2,) and a.dtype == object and a[1].shape == (2,)
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser(description='record OLMo-2 internal state at a readout token per sentence')
    ap.add_argument('--model', help='local model directory')
    ap.add_argument('--stimuli', help='JSON list of records with sid and text')
    ap.add_argument('--out', help='output directory')
    ap.add_argument('--suffix', default='\n', help='suffix after each sentence (backslash escapes allowed)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.model and args.stimuli and args.out):
        ap.error('--model, --stimuli and --out are required')

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    suffix = unescape(args.suffix)
    with open(args.stimuli) as f:
        stim = json.load(f)
    assert isinstance(stim, list) and stim, 'stimuli must be a non-empty JSON list'
    for s in stim:
        assert isinstance(s, dict) and 'sid' in s and 'text' in s, s

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation='eager')
    model = model.to(args.device).eval()
    print(f'suffix {suffix!r} -> ids {tok(suffix, add_special_tokens=False)["input_ids"]}')
    ids, n_text = build_ids(tok, stim[0]['text'], suffix)
    pos = readout_positions(ids, n_text)
    toks = tok.convert_ids_to_tokens(ids)
    print(f'first sentence tokens {toks}; text readout {toks[pos["text"]]!r}, suffix readout {toks[pos["suffix"]]!r}')

    rec = record(model, tok, stim, suffix=suffix, device=args.device)
    save(args.out, rec, stim)
    for k, v in rec.items():
        print(k, v.shape, v.dtype)
    print('wrote', os.path.join(args.out, 'record.npz'), 'and', os.path.join(args.out, 'sentences.json'))


if __name__ == '__main__':
    main()
