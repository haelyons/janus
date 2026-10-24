"""Units of a real language model (OLMo-2-1124-7B) over Pile-10k, as bank cards in bank/text_units.py's format.

A unit is one MLP neuron (layer l, index i) or a unit-norm direction w (length 11008) in the same space: its
activation on a token is entry i of, or the dot product of w with, the input of model.model.layers[l].mlp.down_proj
(act_fn(gate_proj(x)) * up_proj(x)). Position 0 and special tokens never count (NaN in acts.npz, null from serve).

extract (CUDA, torch, transformers) records every unit's activation on every token of Pile-10k into DIR.
build (numpy, pyarrow) writes subject-facing cards (bank/cards/), held-out items (bank/heldout/) and the key
(bank/KEY_olmo.json, orchestrator only: held-out truth, pool windows, simulator items and their truth).
serve (CUDA) answers POST /acts {"unit": name, "texts": [...]} with the unit's activation on each token.

usage: python -I bank/olmo_units.py extract --model DIR --parquet P --out DIR [--batch 8]
       python -I bank/olmo_units.py build --parquet P --out DIR [--select 2]
       python -I bank/olmo_units.py serve --model DIR --units DIR/units.json [--port 8765]
       python -I bank/olmo_units.py --selftest
"""
import argparse
import importlib.util
import json
import random
import re
import sys
import zipfile
import zlib
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numpy.lib import format as npy_format

OUT = Path(__file__).resolve().parent
WIDTH = 11008
RIGHT = 15  # tokens shown after the target token, as in text_units
HELDOUT_PLAN = (('fire', 'surface', 5), ('fire', 'novel', 5), ('no', 'near', 6), ('no', 'any', 4))
SIM_EDGES = (0.05, 0.2, 0.4, 0.6, 0.8)  # act / cmax
SIM_BINS = ('<0.05', '0.05-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '>=0.8')
SIM_COUNTS = (10, 6, 6, 6, 6, 6)
_TU = []


def text_units():
    """bank/text_units.py (window, card format), loaded on first use: it needs tiktoken, extract and serve do not"""
    if not _TU:
        spec = importlib.util.spec_from_file_location('text_units', OUT / 'text_units.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _TU.append(mod)
    return _TU[0]


# ---- units and documents --------------------------------------------------------------------
def draw_units(n_neurons=16, n_dirs=8, seed=33, width=WIDTH):
    rng = np.random.default_rng(seed)
    units = []
    for j in range(n_neurons):
        layer = int(rng.integers(4, 28))
        index = int(rng.integers(0, width))
        units.append({'name': f'olmo:L{layer}:n{index}', 'kind': 'neuron', 'layer': layer, 'index': index, 'w': None})
    for j in range(n_dirs):
        layer = int(rng.integers(4, 28))
        w = rng.standard_normal(width)
        units.append({'name': f'olmo:L{layer}:d{j}', 'kind': 'direction', 'layer': layer, 'index': None,
                      'w': w / np.linalg.norm(w)})
    return units


def unit_json(u):
    """a unit record with w as a list"""
    return {**u, 'w': None if u['w'] is None else [float(x) for x in u['w']]}


def card_id(name):
    return 'O-' + str(zlib.crc32(name.encode()) % 900 + 100)


def doc_parts(n):
    """crc32('olmo:<d>') % 20: 0-9 card, 10-12 held-out, 13-15 pool, 16-19 test"""
    parts = {'card': [], 'heldout': [], 'pool': [], 'test': []}
    for d in range(n):
        r = zlib.crc32(f'olmo:{d}'.encode()) % 20
        parts['card' if r < 10 else 'heldout' if r < 13 else 'pool' if r < 16 else 'test'].append(d)
    return parts


def read_texts(parquet):
    return pq.read_table(parquet, columns=['text']).column('text').to_pylist()


def encode(tokenizer, text, maxtok=1024):
    """tokenize(), plus which positions count (not position 0, not special tokens: those with an empty offset span)"""
    enc = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True, truncation=True, max_length=maxtok)
    ids = [int(x) for x in enc['input_ids']]
    spans = [(int(s), int(e)) for s, e in enc['offset_mapping']]
    special = [s == e for s, e in spans]
    end = next((e for (s, e), sp in zip(reversed(spans), reversed(special)) if not sp), 0)
    text = text[:end]
    offs, nxt = [0] * len(ids), len(text)
    for j in reversed(range(len(ids))):  # a special token takes the start of the next token, len(text) at the end
        offs[j] = nxt if special[j] else spans[j][0]
        nxt = offs[j]
    counts = [j > 0 and not special[j] for j in range(len(ids))]
    return ids, text, offs, counts


def tokenize(tokenizer, text, maxtok=1024):
    """(ids, text, offs) in text_units' doc convention"""
    return encode(tokenizer, text, maxtok)[:3]


# ---- the model --------------------------------------------------------------------------------
def load_model(model_dir):
    """the model (bf16 on CUDA, eval) and its tokenizer, from a local directory"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, local_files_only=True)
    return model.to('cuda').eval(), tokenizer


def pad_id(tokenizer):
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0


class _Stop(Exception):
    """raised by the deepest needed layer's hook: nothing after it is computed"""


def _recorder(layer, units, got, deepest, device):
    """forward pre-hook on layers[layer].mlp.down_proj: got[k] = unit k's activation (batch x tokens)"""
    import torch
    mine = [(k, u) for k, u in enumerate(units) if u['layer'] == layer]
    w = {k: torch.as_tensor(np.asarray(u['w'], np.float32), device=device) for k, u in mine if u['kind'] == 'direction'}

    def hook(module, inputs):
        x = inputs[0].float()
        for k, u in mine:
            got[k] = x[..., u['index']] if u['kind'] == 'neuron' else x @ w[k]
        if layer == deepest:
            raise _Stop
    return hook


def forward_units(model, encs, units, pad):
    """each unit's activation on every token of each encoded text (one right-padded batch, no gradients) ->
    float32 arrays (tokens x units, in units order), NaN where the position never counts"""
    import torch
    lens = [len(e[0]) for e in encs]
    n = max(lens, default=0)
    if n == 0:
        return [np.zeros((0, len(units)), np.float32) for _ in encs]
    device = next(model.parameters()).device
    ids = torch.full((len(encs), n), pad, dtype=torch.long)
    mask = torch.zeros((len(encs), n), dtype=torch.long)
    for b, e in enumerate(encs):
        ids[b, :lens[b]] = torch.tensor(e[0], dtype=torch.long)
        mask[b, :lens[b]] = 1
    got, deepest = {}, max(u['layer'] for u in units)
    hooks = [model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
        _recorder(layer, units, got, deepest, device)) for layer in sorted({u['layer'] for u in units})]
    try:
        with torch.no_grad():
            model.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
    except _Stop:
        pass
    finally:
        for h in hooks:
            h.remove()
    res = torch.stack([got[k] for k in range(len(units))], -1).cpu().numpy()
    out = []
    for b, e in enumerate(encs):
        a = res[b, :lens[b]].astype(np.float32)
        a[~np.asarray(e[3], bool)] = np.nan
        out.append(a)
    return out


def extract(model_dir, parquet, units, out, batch=8, maxtok=1024):
    """one pass over Pile-10k: out/acts.npz (doc_<d>: tokens x units), out/units.json, out/tokens.json
    (per document: ids, offs, and end, the length of the cut text: text = parquet text[:end])"""
    model, tokenizer = load_model(model_dir)
    recs = [unit_json(u) for u in units]
    texts = read_texts(parquet)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'units.json').write_text(json.dumps(recs))
    pad, tokens = pad_id(tokenizer), []
    with zipfile.ZipFile(out / 'acts.npz', 'w', allowZip64=True) as zf:  # written as np.savez does, one doc at a time
        for b0 in range(0, len(texts), batch):
            encs = [encode(tokenizer, t, maxtok) for t in texts[b0:b0 + batch]]
            for d, e, a in zip(range(b0, b0 + len(encs)), encs, forward_units(model, encs, recs, pad)):
                with zf.open(f'doc_{d}.npy', 'w', force_zip64=True) as f:
                    npy_format.write_array(f, a)
                tokens.append({'ids': e[0], 'offs': e[2], 'end': len(e[1])})
            done = b0 + len(encs)
            if done // 500 > b0 // 500 or done == len(texts):
                print(f'extract: {done}/{len(texts)} documents', flush=True)
    (out / 'tokens.json').write_text(json.dumps(tokens))


# ---- cards and items --------------------------------------------------------------------------
def piece(doc, i):
    """token i's text, cut as text_units.window cuts it"""
    ids, text, offs = doc
    return text[offs[i]:offs[i + 1] if i + 1 < len(ids) else len(text)]


def counting(a):
    """positions whose activation counts"""
    return np.flatnonzero(np.isfinite(a))


def doc_peak(a):
    """(position, value) of a document's maximum over counting positions; (None, -inf) if none count"""
    ok = np.isfinite(a)
    if not ok.any():
        return None, -np.inf
    i = int(np.argmax(np.where(ok, a, -np.inf)))
    return i, float(a[i])


def marks(a, cmax):
    """the card's marks: 0-10 scaled to cmax, only at 1 or above, never where the position does not count"""
    out = []
    for x in a:
        v = int(np.round(x / cmax * 10)) if np.isfinite(x) else 0
        out.append(str(min(10, v)) if v >= 1 else None)
    return out


def first_per_doc(pool, k, used=()):
    """the first k (d, i) of pool from distinct documents, skipping documents in used"""
    seen, out = set(used), []
    for d, i in pool:
        if len(out) >= k:
            break
        if d not in seen:
            seen.add(d)
            out.append((d, i))
    return out


def card(cid, docs, acts, card_docs, rng, left=48):
    """the card for one unit -> (text, cmax, peaks), in text_units.card_text's field format"""
    window = text_units().window
    at = [(d,) + doc_peak(acts[d]) for d in sorted(card_docs)]
    at = [x for x in at if x[1] is not None]
    top = sorted(at, key=lambda x: -x[2])[:12]
    cmax = top[0][2]
    mids = [x for x in at if 0.3 * cmax <= x[2] <= 0.7 * cmax]
    mid = sorted(rng.sample(mids, min(8, len(mids))), key=lambda x: -x[2])
    scale_note = 'Activations are scaled 0-10 relative to the strongest example shown; only tokens at 1 or above are marked.'
    lines = [f'# Unit {cid}', '',
             'Recorded from one unit (a single neuron) in a language model, run over a broad sample of text '
             '(web pages, code, science abstracts, forums, law, Wikipedia).',
             f'Each example is a window of text around the token where the unit is most active in one document. '
             f'Marked tokens look like «token|activation». {scale_note}', '',
             '## Strongest activations (one example per document)']
    for n, (d, i, v) in enumerate(top, 1):
        lines += ['', f'### Example {n}', window(docs[d], i, left, RIGHT, marks(acts[d], cmax))]
    lines += ['', '## Examples from the middle of the activation range (30-70% of the strongest)']
    if not mid:
        lines += ['', '(none found)']
    for n, (d, i, v) in enumerate(mid, 1):
        lines += ['', f'### Middle example {n}', window(docs[d], i, left, RIGHT, marks(acts[d], cmax))]
    peaks = [piece(docs[d], i).strip().lower() for d, i, v in top]
    return '\n'.join(lines) + '\n', cmax, peaks


def heldout_items(docs, acts, ho_docs, cmax, peaks, rng, left=48, name='unit'):
    """20 held-out items -> (items, truth) as text_units.heldout returns them; a tag's shortfall is taken from the
    other tag of its kind (items keep the tag they were found under); a kind short overall raises SystemExit"""
    window = text_units().window
    peakset, pools = set(peaks), {}
    for kind, tag, _ in HELDOUT_PLAN:
        pool = []
        for d in ho_docs:
            a = acts[d]
            idx = counting(a)
            if kind == 'fire':
                for i in idx[a[idx] >= 0.3 * cmax].tolist():
                    if (piece(docs[d], i).strip().lower() in peakset) == (tag == 'surface'):
                        pool.append((d, i))
            else:
                for i in idx[a[idx] <= 0.05 * cmax].tolist():
                    if tag == 'near':
                        hit = piece(docs[d], i).strip().lower() in peakset
                    else:
                        hit = rng.random() < 0.01
                    if hit:
                        pool.append((d, i))
        rng.shuffle(pool)
        pools[tag] = pool
    chosen = []
    for kind in ('fire', 'no'):
        (t1, q1), (t2, q2) = [(t, q) for k, t, q in HELDOUT_PLAN if k == kind]
        c1, c2 = first_per_doc(pools[t1], q1), first_per_doc(pools[t2], q2)  # one item per document
        s1, s2 = q1 - len(c1), q2 - len(c2)
        if s1:
            c2 = first_per_doc(pools[t2], q2 + s1)
        if s2:
            c1 = first_per_doc(pools[t1], q1 + s2)
        if len(c1) + len(c2) < q1 + q2:
            raise SystemExit(f'{name}: only {len(c1) + len(c2)} held-out {kind} items (need {q1 + q2})')
        chosen += [(d, i, kind == 'fire', t1) for d, i in c1] + [(d, i, kind == 'fire', t2) for d, i in c2]
    rng.shuffle(chosen)
    return [{'id': n, 'text': window(docs[d], i, left, RIGHT, target=True)} for n, (d, i, _, _) in enumerate(chosen, 1)], \
           [{'id': n, 'fires': f, 'tag': t, 'act': round(float(acts[d][i]), 3)} for n, (d, i, f, t) in enumerate(chosen, 1)]


def pool_windows(docs, acts, pool_docs, cmax, rng, n_fire=8, n_any=7, left=48):
    """n_fire windows around tokens at >= 0.3 x cmax (one per document), n_any around uniformly chosen counting
    tokens of the other pool documents; the card's marks; shuffled"""
    window = text_units().window
    fire = []
    for d in pool_docs:
        idx = counting(acts[d])
        fire += [(d, i) for i in idx[acts[d][idx] >= 0.3 * cmax].tolist()]
    rng.shuffle(fire)
    chosen = first_per_doc(fire, n_fire)
    used = {d for d, _ in chosen}
    others = [d for d in pool_docs if d not in used]
    idxs = [counting(acts[d]) for d in others]
    ends = np.cumsum([len(x) for x in idxs])
    total = int(ends[-1]) if len(ends) else 0
    for p in rng.sample(range(total), min(n_any, total)):
        j = int(np.searchsorted(ends, p, side='right'))
        chosen.append((others[j], int(idxs[j][p - (int(ends[j]) - len(idxs[j]))])))
    wins = [window(docs[d], i, left, RIGHT, marks(acts[d], cmax)) for d, i in chosen]
    rng.shuffle(wins)
    return wins


def sim_items(docs, acts, test_docs, cmax, rng, left=48):
    """40 test tokens, at most one per document, by act / cmax bin (SIM_BINS, SIM_COUNTS); a bin's shortfall is
    taken from the next lower bin's tokens -> (items, truth {"id", "act", "bin"})"""
    window = text_units().window
    pools = [[] for _ in SIM_BINS]
    for d in test_docs:
        a = acts[d]
        idx = counting(a)
        for i, k in zip(idx.tolist(), np.searchsorted(SIM_EDGES, a[idx] / cmax, side='right').tolist()):
            pools[k].append((d, i))
    chosen, used, carry = [], set(), 0
    for k in reversed(range(len(SIM_BINS))):
        rng.shuffle(pools[k])
        got = first_per_doc(pools[k], SIM_COUNTS[k] + carry, used)
        used.update(d for d, _ in got)
        chosen += [(d, i, k) for d, i in got]
        carry = SIM_COUNTS[k] + carry - len(got)
    rng.shuffle(chosen)
    return [{'id': n, 'text': window(docs[d], i, left, RIGHT, target=True)} for n, (d, i, _) in enumerate(chosen, 1)], \
           [{'id': n, 'act': round(float(acts[d][i]), 3), 'bin': SIM_BINS[k]} for n, (d, i, k) in enumerate(chosen, 1)]


def build(parquet, out, model_dir=None, select=2):
    """cards, held-out items and key entries from extract's outputs in out (model_dir is not needed)"""
    out = Path(out)
    units = json.loads((out / 'units.json').read_text())
    toks = json.loads((out / 'tokens.json').read_text())
    raw = read_texts(parquet)
    assert len(toks) == len(raw), (len(toks), len(raw))
    docs = [(t['ids'], raw[d][:t['end']], t['offs']) for d, t in enumerate(toks)]
    with np.load(out / 'acts.npz') as z:
        mats = [z[f'doc_{d}'] for d in range(len(docs))]
    parts = doc_parts(len(docs))
    key_path = OUT / 'KEY_olmo.json'
    key = json.loads(key_path.read_text()) if key_path.exists() else {}
    (OUT / 'cards').mkdir(exist_ok=True)
    (OUT / 'heldout').mkdir(exist_ok=True)
    taken = 0
    for k, u in enumerate(units):
        name, cid = u['name'], card_id(u['name'])
        acts = [m[:, k] for m in mats]
        rng = random.Random(zlib.crc32(name.encode()))
        text, cmax, peaks = card(cid, docs, acts, parts['card'], rng)
        n = sum(doc_peak(acts[d])[1] >= 0.3 * cmax for d in parts['card'])
        if n < 40:
            print(f'{name}: not usable: {n} card documents reach 0.3 x card max (need 40)', flush=True)
            continue
        try:
            items, truth = heldout_items(docs, acts, parts['heldout'], cmax, peaks, rng, name=name)
        except SystemExit as e:
            print(f'{name}: not usable: {e}', flush=True)
            continue
        if u['kind'] == 'neuron' and taken >= select:
            print(f'{name}: usable, not taken ({select} neurons already taken)', flush=True)
            continue
        if key.get(cid, {}).get('unit', name) != name:
            raise SystemExit(f'{cid}: already the id of {key[cid]["unit"]}, also wanted by {name}')
        windows = pool_windows(docs, acts, parts['pool'], cmax, rng)
        sim, sim_truth = sim_items(docs, acts, parts['test'], cmax, rng)
        (OUT / 'cards' / f'{cid}.md').write_text(text)
        (OUT / 'heldout' / f'{cid}.json').write_text(json.dumps(items, ensure_ascii=False, indent=1))
        key[cid] = {'unit': name, 'kind': u['kind'], 'format': 'field', 'card_max': cmax, 'heldout': truth,
                    'pool': windows, 'sim': sim, 'sim_truth': sim_truth, 'peaks': peaks}
        key_path.write_text(json.dumps(key, indent=1))
        taken += u['kind'] == 'neuron'
        print(f'{name}: usable, taken as {cid}', flush=True)


# ---- serving ----------------------------------------------------------------------------------
class UnitServer:
    """the model, loaded once, giving one unit's activation on new texts"""

    def __init__(self, model_dir, units_json):
        self.model, self.tokenizer = load_model(model_dir)
        self.pad = pad_id(self.tokenizer)
        self.units = {u['name']: u for u in json.loads(Path(units_json).read_text())}

    def acts(self, name, texts, maxtok=1024):
        encs = [encode(self.tokenizer, t, maxtok) for t in texts]
        arrs = forward_units(self.model, encs, [self.units[name]], self.pad)
        return [{'text': e[1], 'offs': e[2], 'acts': [float(x) if np.isfinite(x) else None for x in a[:, 0]]}
                for e, a in zip(encs, arrs)]


def serve(model_dir, units_json, port=8765):
    srv = UnitServer(model_dir, units_json)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != '/acts':
                return self.reply(404, {'error': 'POST /acts only'})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                name, texts = req['unit'], req['texts']
            except (ValueError, KeyError, TypeError):
                return self.reply(400, {'error': 'expected {"unit": name, "texts": [str, ...]}'})
            if not isinstance(name, str) or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                return self.reply(400, {'error': 'expected {"unit": name, "texts": [str, ...]}'})
            if name not in srv.units:
                return self.reply(400, {'error': f'unknown unit {name!r}'})
            self.reply(200, {'results': srv.acts(name, texts)})

    print(f'serving {len(srv.units)} units on http://127.0.0.1:{port}/acts', flush=True)
    HTTPServer(('127.0.0.1', port), Handler).serve_forever()


# ---- selftest ---------------------------------------------------------------------------------
class FakeTokenizer:
    """stands in for a fast HF tokenizer: whitespace-led words, one special token first (and one last with eos)"""

    def __init__(self, eos=False):
        self.tail = 1 if eos else 0

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=True, truncation=True, max_length=1024):
        spans = [m.span() for m in re.finditer(r'\s*\S+', text)][:max_length - 1 - self.tail]
        ids = [1 + zlib.crc32(text[s:e].encode()) % 1000 for s, e in spans]
        return {'input_ids': [0] + ids + [0] * self.tail, 'offset_mapping': [(0, 0)] + spans + [(0, 0)] * self.tail}


def _synth(plants, n_docs=240, n_words=80, seed=0):
    """synthetic docs (filler words carry their document number) and activations, NaN at the special token;
    plants: d -> [(token position, word, activation)]"""
    rs = np.random.default_rng(seed)
    filler = ('alpha', 'beta', 'gamma', 'delta', 'kappa', 'sigma')
    docs, acts = [], []
    for d in range(n_docs):
        words = [f'{filler[k]}{d}' for k in rs.integers(0, len(filler), n_words)]
        a = rs.uniform(0, 0.02, n_words + 1)
        for i, word, v in plants.get(d, ()):
            words[i - 1], a[i] = word, v
        doc = tokenize(FakeTokenizer(), ''.join(' ' + w for w in words))
        assert len(doc[0]) == n_words + 1
        a[0] = np.nan
        docs.append(doc)
        acts.append(a)
    return docs, acts


def selftest():
    tu = text_units()
    # draw_units
    u1, u2 = draw_units(), draw_units()
    names = [u['name'] for u in u1]
    assert names == [u['name'] for u in u2] and len(names) == 24
    assert [u['kind'] for u in u1] == ['neuron'] * 16 + ['direction'] * 8
    assert all(np.array_equal(a['w'], b['w']) for a, b in zip(u1[16:], u2[16:]))
    for u in u1[:16]:
        assert u['name'] == f"olmo:L{u['layer']}:n{u['index']}" and 4 <= u['layer'] < 28 and 0 <= u['index'] < WIDTH
    for j, u in enumerate(u1[16:]):
        assert u['name'] == f"olmo:L{u['layer']}:d{j}" and 4 <= u['layer'] < 28 and u['index'] is None
        assert len(u['w']) == WIDTH and abs(np.linalg.norm(u['w']) - 1) < 1e-9
    assert [u['name'] for u in draw_units(seed=34)] != names
    assert json.loads(json.dumps(unit_json(u1[-1])))['w'] == [float(x) for x in u1[-1]['w']]
    assert re.fullmatch(r'O-\d{3}', card_id(names[0]))
    parts = doc_parts(1000)
    assert sorted(sum(parts.values(), [])) == list(range(1000)) and all(parts.values())

    # tokenize: special first gets the next start, text cut at the last non-special token's end
    t = ' one two, three four  \n'
    ids, text, offs = tokenize(FakeTokenizer(), t, maxtok=4)
    assert (len(ids), text, offs) == (4, ' one two, three', [0, 0, 4, 9]), (ids, text, offs)
    doc = (ids, text, offs)
    assert tu.window(doc, 2, 48, 15) == text
    assert tu.window(doc, 2, 48, 15, target=True) == ' one⟦ two,⟧ three'
    assert tu.window(doc, 3, 1, 15, [None, None, '7', None]) == '…« two,|7» three'
    ids, text, offs, counts = encode(FakeTokenizer(eos=True), t)
    assert (text, offs) == (' one two, three four', [0, 0, 4, 9, 15, 20]), (text, offs)
    assert counts == [False, True, True, True, True, False] and tu.window((ids, text, offs), 4, 48, 15) == text

    # synthetic corpus: 0-99 card, 100-149 held-out, 150-169 pool, 170-239 test
    perm = np.random.default_rng(5).permutation(100)
    val = {d: 0.1 + 0.9 * int(perm[d]) / 99 for d in range(100)}
    top_d = sorted(val, key=lambda d: -val[d])[:12]
    base = {d: [(10, f'w{d}', val[d])] for d in range(100)}
    base[top_d[0]] = base[top_d[0]] + [(20, 'marked', 0.42), (21, 'faint', 0.04)]

    def ho(n_surface, n_novel, n_near):
        p = {100 + j: [(30, f'w{top_d[0]}', 0.8)] for j in range(n_surface)}
        p.update({110 + j: [(30, f'novel{110 + j}', 0.6)] for j in range(n_novel)})
        p.update({120 + j: [(30, f'w{top_d[1]}', 0.0)] for j in range(n_near)})
        return p

    plants = {**base, **ho(10, 10, 10)}
    plants.update({d: [(30, f'fire{d}', 2.0)] for d in range(150, 158)})
    for lo, hi, v in ((170, 173, 0.9), (173, 185, 0.7), (185, 191, 0.5), (191, 197, 0.3), (197, 203, 0.1)):
        plants.update({d: [(40, f'hi{d}', v)] for d in range(lo, hi)})
    docs, acts = _synth(plants)

    # card: top = highest-maximum documents strongest first, never at the NaN position; marks scale; middle range
    text, cmax, peaks = card('O-123', docs, acts, list(range(100)), random.Random(1))
    assert peaks == [f'w{d}' for d in top_d], peaks
    assert cmax == max(val.values()) and text.startswith('# Unit O-123\n\nRecorded from one unit')
    top_sec, mid_sec = text.split('## Examples from the middle')
    assert top_sec.count('### Example ') == 12 and mid_sec.count('### Middle example ') == 8
    assert [int(x) for x in re.findall(r'« w(\d+)\|', top_sec)] == top_d
    assert f'« w{top_d[0]}|10»' in top_sec and '« marked|4»' in top_sec and '« faint' not in text and '«|' not in text
    mid = [int(x) for x in re.findall(r'« w(\d+)\|\d+»', mid_sec)]
    assert len(mid) == 8 and all(0.3 * cmax <= val[d] <= 0.7 * cmax for d in mid), mid
    assert [val[d] for d in mid] == sorted((val[d] for d in mid), reverse=True)

    # held-out: plan counts and tags, shortfall from the other tag of the kind, a kind short overall raises
    ho_docs = list(range(100, 150))

    def check_heldout(items, truth, want):
        assert [x['id'] for x in items] == [t['id'] for t in truth] == list(range(1, 21))
        assert Counter(t['tag'] for t in truth) == Counter(want), Counter(t['tag'] for t in truth)
        for x, t in zip(items, truth):
            tgt = re.findall('⟦(.*?)⟧', x['text'])
            assert len(tgt) == 1 and tgt[0] and np.isfinite(t['act']), x
            s = tgt[0].strip().lower()
            assert t['fires'] == (t['tag'] in ('surface', 'novel'))
            assert (t['act'] >= 0.3 * cmax) if t['fires'] else (t['act'] <= 0.05 * cmax), t
            assert t['tag'] not in ('surface', 'near') or s in peaks, (t, s)
            assert t['tag'] != 'novel' or s not in peaks, (t, s)

    check_heldout(*heldout_items(docs, acts, ho_docs, cmax, peaks, random.Random(2)),
                  {'surface': 5, 'novel': 5, 'near': 6, 'any': 4})
    d2, a2 = _synth({**base, **ho(3, 10, 2)})
    check_heldout(*heldout_items(d2, a2, ho_docs, cmax, peaks, random.Random(2)),
                  {'surface': 3, 'novel': 7, 'near': 2, 'any': 8})
    d3, a3 = _synth({**base, **ho(3, 4, 10)})
    try:
        heldout_items(d3, a3, ho_docs, cmax, peaks, random.Random(2), name='olmo:test')
        raise AssertionError('a fire shortfall of 3 did not raise')
    except SystemExit as e:
        assert 'olmo:test' in str(e) and 'fire' in str(e), e

    # pool windows: 8 fire windows (one per document, marks capped at 10), 7 unmarked from other documents
    wins = pool_windows(docs, acts, list(range(150, 170)), cmax, random.Random(3))
    marked = [w for w in wins if '«' in w]
    assert len(wins) == 15 and len(marked) == 8 and '«|' not in ''.join(wins)
    assert all(re.search(r'« fire\d+\|10»', w) for w in marked)
    assert len({re.search(r'« fire(\d+)\|', w).group(1) for w in marked}) == 8

    # sim items: bins, the shortfall of the top bin taken from the next lower, one per document, never NaN
    items, truth = sim_items(docs, acts, list(range(170, 240)), cmax, random.Random(4))
    assert [x['id'] for x in items] == [t['id'] for t in truth] == list(range(1, 41))
    assert Counter(t['bin'] for t in truth) == Counter(
        {'<0.05': 10, '0.05-0.2': 6, '0.2-0.4': 6, '0.4-0.6': 6, '0.6-0.8': 9, '>=0.8': 3}), Counter(t['bin'] for t in truth)
    tgts = [re.findall('⟦(.*?)⟧', x['text']) for x in items]
    assert all(len(t) == 1 and t[0] for t in tgts)
    src = {int(re.search(r'(\d+)$', t[0]).group(1)) for t in tgts}
    assert len(src) == 40 and all(170 <= d < 240 for d in src), src
    for t in truth:
        assert np.isfinite(t['act'])
        assert SIM_BINS[int(np.searchsorted(SIM_EDGES, t['act'] / cmax, side='right'))] == t['bin'], t
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description='OLMo-2-1124-7B units over Pile-10k as bank cards')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--model', required=True)
    p.add_argument('--parquet', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batch', type=int, default=8)
    p = sub.add_parser('build')
    p.add_argument('--parquet', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--select', type=int, default=2)
    p = sub.add_parser('serve')
    p.add_argument('--model', required=True)
    p.add_argument('--units', required=True)
    p.add_argument('--port', type=int, default=8765)
    a = ap.parse_args()
    if a.cmd == 'extract':
        extract(a.model, a.parquet, draw_units(), a.out, batch=a.batch)
    elif a.cmd == 'build':
        build(a.parquet, a.out, select=a.select)
    else:
        serve(a.model, a.units, a.port)


if __name__ == '__main__':
    main()
