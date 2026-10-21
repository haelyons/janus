"""Do readers' claims follow what an invariance profile shows? (spec: authors/spec_profilecheck.md)

Units: 3 channels drawn with default_rng(22) from each of mixed3a, mixed3b, mixed4b, mixed4d, mixed4e and mixed5a
(skipping channels studied before), plus the 9 round-21 units of runs/lookviews/readings_all.json, whose base and P
readings already exist. Random units: values, cards and card text as tools/openmethod.py (values computed on CUDA when
available and checked against openmethod's own CPU computation), then readings under lookviews' conditions base and P,
3 samples each. Each reading's explanation goes to one text-only call that says, for 10 properties, whether it claims
the unit's response "depends" on it, is "unaffected" by it, or does not say ("not_said"). Each claim gets a verdict
against the unit's percentile p for the matching transform in its lookviews invariance profile. Outputs:
runs/profilecheck/.

usage: python -I tools/profilecheck.py [--selftest]
"""
import argparse
import gc
import json
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir
import battery  # noqa: E402
import bundlemax  # noqa: E402
import claude_call  # noqa: E402
import lookviews  # noqa: E402
import openmethod  # noqa: E402
import run_subjects  # noqa: E402

OUT = J / 'runs/profilecheck'
LV = lookviews.OUT
NCH = openmethod.NCH
BATCH, N_CHECK, WORKERS = openmethod.BATCH, 64, 6
SEED, PER_LAYER = 22, 3
RANDOM_LAYERS = ('mixed3a', 'mixed3b', 'mixed4b', 'mixed4d', 'mixed4e', 'mixed5a')
EXCLUDE = {
    'mixed3a': {136, 108, 132, 88, 110, 180, 153, 186, 86, 117, 112, 70, 106, 113, 147, 81, 145, 71},
    'mixed3b': {379, 385, 343, 342, 388, 340, 330, 349, 406, 324, 380, 404, 303},
    'mixed4b': {237, 491, 373, 443, 431, 429, 384, 409, 418},
    'mixed4d': {409, 417, 241, 277, 352, 455, 167, 295, 174, 393},
    'mixed4e': {55, 809, 804, 703, 315, 700},
    'mixed5a': {719},
}
N_ROUND21 = 9
GROUPS = ('random', 'round21')
CONDITIONS = ('base', 'P')
SAMPLES = (1, 2, 3)
READINGS = [f'{c}_s{k}' for c in CONDITIONS for k in SAMPLES]
PROPERTIES = ('colour', 'fine_detail', 'coarse_shading', 'shape', 'arrangement', 'orientation', 'upright',
              'left_right', 'inner_texture', 'polarity')
CLAIM_VALUES = ('depends', 'unaffected', 'not_said')
VERDICTS = ('supported', 'unremarkable', 'contradicted')
PROPERTY_TRANSFORM = {'colour': 'grayscale', 'fine_detail': 'low_pass', 'coarse_shading': 'high_pass',
                      'shape': 'phase_scramble', 'arrangement': 'patch_shuffle_4x4', 'orientation': 'rotate_90',
                      'upright': 'rotate_180', 'left_right': 'mirror', 'inner_texture': 'edge_map',
                      'polarity': 'luminance_polarity_inversion'}
P_LOW, P_HIGH = 20, 80
GAP = 0.25

CLAIM_PROMPT = (
    'Here is an explanation of what one unit (neuron) of an image classification network responds to.\n\n'
    '{text}\n\n'
    'For each property below, say whether the explanation claims that the unit\'s response depends on it ("depends"), '
    'claims that the response is unaffected by it ("unaffected"), or does not say ("not_said"). Count only what the '
    'explanation states or clearly implies about the unit\'s response, not descriptions of the photos.\n'
    '- colour: colour or hue\n'
    '- fine_detail: fine detail and sharp edges\n'
    '- coarse_shading: smooth shading and large-scale brightness gradients\n'
    '- shape: the spatial structure or shape of what is shown\n'
    '- arrangement: the overall arrangement of parts across the image\n'
    '- orientation: orientation in general (which way edges or objects are tilted)\n'
    '- upright: being upright rather than upside down\n'
    '- left_right: left-right direction (which way something faces or points)\n'
    '- inner_texture: surface texture and shading inside regions, beyond outlines\n'
    '- polarity: whether something is light on dark or dark on light\n'
    'Reply with only JSON with exactly these keys: {"colour": ..., "fine_detail": ..., ...}')


def cond_of(rid):
    """'<condition>_s<k>' -> condition"""
    return rid.rsplit('_s', 1)[0]


# ---------- units ----------

def draw_units(nch, exclude, seed=22):
    """nch: ordered {layer: n_channels}; exclude: {layer: set of channels}. One generator default_rng(seed) for all
    layers, in nch order: rng.integers(n_channels) repeatedly, skipping excluded and already drawn channels, until
    PER_LAYER are kept. -> [(layer, channel)] in draw order. ValueError if a layer has too few channels left."""
    rng = np.random.default_rng(seed)
    out = []
    for layer, n in nch.items():
        n = int(n)
        ex = {int(c) for c in exclude.get(layer, ())}
        if n - len({c for c in ex if 0 <= c < n}) < PER_LAYER:
            raise ValueError(f'{layer}: fewer than {PER_LAYER} channels outside the exclusions')
        kept = []
        while len(kept) < PER_LAYER:
            c = int(rng.integers(n))
            if c in ex or c in kept:
                continue
            kept.append(c)
        out += [(layer, c) for c in kept]
    return out


# ---------- activations (openmethod's definition, on CUDA when available) ----------

def exact_cuda():
    """full fp32 on the GPU (no TF32), so the values can match the CPU computation"""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_net(dev):
    import torch
    torch.set_num_threads(4)
    m = bundlemax._bank().load_model(openmethod.WEIGHTS)
    for q in m.parameters():
        q.requires_grad_(False)
    return m.to(dev)


def forward_max_on(model, u, ids, dev, tag=''):
    """openmethod.forward_max with the model and inputs on dev: channel u['chan'] of the output of the module named
    u['layer'] moved to the CPU, then openmethod.spatial_max (max, its row and col, map size); batches of BATCH"""
    import torch
    Stop = bundlemax._bank().Stop
    got = {}

    def stop(_m, _i, out):
        assert out.shape[1] == NCH[u['layer']], tuple(out.shape)
        got['m'] = out[:, u['chan']].detach().cpu().numpy().astype(np.float32)
        raise Stop

    vals, rows, cols, sizes = [], [], [], set()
    h = dict(model.named_modules())[u['layer']].register_forward_hook(stop)
    try:
        for b, s in enumerate(range(0, len(ids), BATCH)):
            x = bundlemax.net_input(ids[s:s + BATCH]).to(dev)
            got.clear()
            try:
                with torch.no_grad():
                    model(x)
            except Stop:
                pass
            v, r, c, n = openmethod.spatial_max(got['m'])
            vals.append(v)
            rows.append(r)
            cols.append(c)
            sizes.add(n)
            if b % 20 == 19:
                print(f'  {tag} {min(s + BATCH, len(ids))}/{len(ids)}', flush=True)
    finally:
        h.remove()
    assert len(sizes) == 1, sizes
    return np.concatenate(vals), np.concatenate(rows), np.concatenate(cols), sizes.pop()


def unit_values(u, C, H, udir, get_model, get_cpu_model, dev):
    """values.npz in openmethod's format. Cached: loaded through openmethod.unit_values (its own cache check). Else
    computed on dev; before saving, the first N_CHECK card-pool photos are checked against openmethod.forward_max on
    the CPU model (np.allclose, rtol 1e-3, and the same map size); stop if they differ."""
    f = udir / 'values.npz'
    if f.exists():
        return openmethod.unit_values(u, C, H, udir, None)
    v, r, c, size = forward_max_on(get_model(), u, np.concatenate([C, H]), dev, tag=f'{u["name"]} ({dev})')
    v0, _, _, size0 = openmethod.forward_max(get_cpu_model(), u, C[:N_CHECK], tag=f'{u["name"]} (cpu check)')
    k = len(v0)
    if size0 != size or not np.allclose(v[:k], v0, rtol=1e-3):
        dmax = float(np.max(np.abs(v[:k].astype(np.float64) - v0))) if size0 == size else float('nan')
        raise SystemExit(f'{u["name"]}: {dev} values differ from openmethod on the CPU on the first {k} card-pool '
                         f'photos (map {size} vs {size0}, max abs diff {dmax:.3g}); stopping')
    n = len(C)
    d = {'unit': np.array(u['name']), 'C': C, 'H': H, 'vC': v[:n], 'vH': v[n:], 'rC': r[:n], 'cC': c[:n],
         'rH': r[n:], 'cH': c[n:], 'size': np.array(size)}
    tmp = udir / 'values.tmp.npz'
    np.savez_compressed(tmp, **d)
    tmp.replace(f)
    return d


# ---------- readings (conditions base and P) ----------

def reading_contents(content, profile):
    """{'base', 'P'} exactly as lookviews.condition_contents builds them from openmethod's card content; the crops
    view is not used by either"""
    cs = lookviews.condition_contents(content[0]['text'], content[1], content[2], None, profile, None)
    return {c: cs[c] for c in CONDITIONS}


def readings(name, contents, udir, call=None):
    """lookviews.readings for CONDITIONS only: every (condition, sample) reading as openmethod.get_reading with
    system prompt '', up to WORKERS in parallel, cached in readings.json keyed '<condition>_s<k>'; a failed call
    (is_error) is not cached"""
    path = udir / 'readings.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    sha = {c: openmethod.sha256(json.dumps(contents[c], sort_keys=True)) for c in CONDITIONS}
    jobs = []
    for c in CONDITIONS:
        for k in SAMPLES:
            key = f'{c}_s{k}'
            if key not in cache:
                jobs.append((key, c, k))
            elif cache[key]['content_sha256'] != sha[c]:
                raise SystemExit(f'{path}: cached {key} was made from different content; stopping')
    if jobs:
        print(name, 'readings', [j[0] for j in jobs], flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(openmethod.get_reading, '', contents[c], call): (key, c, k) for key, c, k in jobs}
        for f in as_completed(futs):
            key, c, k = futs[f]
            try:
                rec = f.result()
            except Exception as err:  # not cached, so a rerun tries again
                print(f'{name} {key} failed: {err}', flush=True)
                continue
            cache[key] = {'unit': name, 'reading': key, 'condition': c, 'sample': k, 'model': claude_call.MODEL,
                          'system_prompt': '', 'content_sha256': sha[c], **rec}
            lookviews.save_json(cache, path)
    return cache


# ---------- claim extraction ----------

def claims_prompt(text):
    return CLAIM_PROMPT.replace('{text}', text)


def parse_claims(text):
    """the last JSON object in the reply that has the 10 property keys; valid only with exactly those keys, each
    value one of CLAIM_VALUES -> {property: value}, else None"""
    if not isinstance(text, str):
        return None
    o = bundlemax.last_obj(text, PROPERTIES)
    if o is None or set(o) != set(PROPERTIES):
        return None
    if not all(isinstance(o[k], str) and o[k] in CLAIM_VALUES for k in PROPERTIES):
        return None
    return {k: o[k] for k in PROPERTIES}


def _claims_call(content):
    return claude_call.converse('', [content])[0]


def get_claims(text, call=None):
    """one fresh single-turn text-only call, parsed by parse_claims; one retry in a fresh conversation on an invalid
    reply, then missing. An is_error reply raises (not cached, so a rerun tries again)."""
    call = call or _claims_call
    content = [{'type': 'text', 'text': claims_prompt(text)}]
    replies = []
    for a in range(2):
        r = call(content)
        if r.get('is_error'):
            raise RuntimeError(f'error reply, not cached: {str(r.get("text"))[:80]}')
        replies.append(r['text'])
        o = parse_claims(r['text'])
        if o is not None:
            return {'valid': True, 'claims': o, 'reply': r['text'], 'replies': replies, 'attempts': a + 1}
    return {'valid': False, 'claims': None, 'reply': replies[-1], 'replies': replies, 'attempts': 2}


def unit_claims(name, texts, udir, call=None):
    """claims of every reading in texts ({reading id: explanation}), up to WORKERS in parallel, cached in claims.json
    keyed by reading id with the explanation's sha256; a changed explanation stops; a failed call is not cached"""
    path = udir / 'claims.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    sha = {rid: openmethod.sha256(t) for rid, t in texts.items()}
    jobs = []
    for rid in texts:
        if rid not in cache:
            jobs.append(rid)
        elif cache[rid]['text_sha256'] != sha[rid]:
            raise SystemExit(f'{path}: cached {rid} was made from a different explanation; stopping')
    if jobs:
        print(name, 'claims', jobs, flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(get_claims, texts[rid], call): rid for rid in jobs}
        for f in as_completed(futs):
            rid = futs[f]
            try:
                rec = f.result()
            except Exception as err:  # not cached, so a rerun tries again
                print(f'{name} {rid} claims failed: {err}', flush=True)
                continue
            cache[rid] = {'unit': name, 'reading': rid, 'condition': cond_of(rid), 'model': claude_call.MODEL,
                          'text_sha256': sha[rid], **rec}
            lookviews.save_json(cache, path)
    return cache


# ---------- agreement and summary ----------

def verdict(claim, p):
    """'depends': supported if p <= 20, contradicted if p >= 80; 'unaffected': supported if p >= 80, contradicted if
    p <= 20; otherwise unremarkable (also for an undefined p, NaN, where neither condition holds). None for
    'not_said'."""
    if claim == 'not_said':
        return None
    if claim == 'depends':
        return 'supported' if p <= P_LOW else 'contradicted' if p >= P_HIGH else 'unremarkable'
    if claim == 'unaffected':
        return 'supported' if p >= P_HIGH else 'contradicted' if p <= P_LOW else 'unremarkable'
    raise ValueError(f'bad claim {claim!r}')


def summarize(records):
    """records: [{"group", "condition", "claims": {property: value}, "pct": {transform: p}}]; records whose claims
    are None are not readings with claims and are skipped. -> {condition: {group: {readings, claims_per_reading,
    <verdict>_per_reading, <verdict>_share}}}, groups in first-seen order and then "all"; shares None without claims"""
    tally = {}
    for r in records:
        cl = r.get('claims')
        if cl is None:
            continue
        g, c = r['group'], r['condition']
        if g == 'all':
            raise ValueError("'all' is not a group name")
        t = {'claims': 0, **{v: 0 for v in VERDICTS}}
        for prop, value in cl.items():
            if value == 'not_said':
                continue
            t['claims'] += 1
            t[verdict(value, r['pct'][PROPERTY_TRANSFORM[prop]])] += 1
        for key in (g, 'all'):
            x = tally.setdefault(c, {}).setdefault(key, {'readings': 0, 'claims': 0, **{v: 0 for v in VERDICTS}})
            x['readings'] += 1
            for k, n in t.items():
                x[k] += n
    out = {}
    for c, groups in tally.items():
        out[c] = {}
        for g in [g for g in groups if g != 'all'] + ['all']:
            x = groups[g]
            n, k = x['readings'], x['claims']
            s = {'readings': n, 'claims_per_reading': k / n}
            s.update({f'{v}_per_reading': x[v] / n for v in VERDICTS})
            s.update({f'{v}_share': (x[v] / k if k else None) for v in VERDICTS})
            out[c][g] = s
    return out


def mid_profile(pct):
    """all 10 percentiles in the open interval (20, 80)"""
    return all(P_LOW < pct[t] < P_HIGH for t in battery.TRANSFORMS)


def share_gap(s):
    """supported share - contradicted share of one summary cell; None without claims"""
    if s is None or s['supported_share'] is None:
        return None
    return s['supported_share'] - s['contradicted_share']


def decide_gap(g):
    """Neutral label on supported share - contradicted share: > 0.25 SUPPORTED_MORE, < -0.25 CONTRADICTED_MORE,
    else SIMILAR; UNDEFINED without claims."""
    if g is None or not np.isfinite(g):
        return 'UNDEFINED'
    return 'SUPPORTED_MORE' if g > GAP else 'CONTRADICTED_MORE' if g < -GAP else 'SIMILAR'


# ---------- report ----------

def fmt(x, nd=3):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:.{nd}f}'


def fmt_signed(x):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:+.3f}'


def clean(x):
    """a float, or None if undefined (JSON has no NaN)"""
    return None if x is None or not np.isfinite(x) else float(x)


def claim_items(claims, pct):
    """[(property, value, verdict)] of the claims (not 'not_said'), in PROPERTIES order"""
    return [(p, claims[p], verdict(claims[p], pct[PROPERTY_TRANSFORM[p]])) for p in PROPERTIES
            if claims[p] != 'not_said']


def claims_text(claims, pct):
    items = claim_items(claims, pct)
    return ' '.join(f'{p}:{v}:{vd}' for p, v, vd in items) if items else '(no claims)'


def pct_line(pct):
    return 'percentiles: ' + ', '.join(f'{t} {fmt(pct[t], 1)}' for t in battery.TRANSFORMS)


def cell_line(g, s):
    if s is None:
        return f'{g:<8} no readings with claims'
    gap = share_gap(s)
    return (f'{g:<8} readings {s["readings"]}  claims/reading {fmt(s["claims_per_reading"])}  per reading: supported '
            f'{fmt(s["supported_per_reading"])} unremarkable {fmt(s["unremarkable_per_reading"])} contradicted '
            f'{fmt(s["contradicted_per_reading"])}  shares of claims: supported {fmt(s["supported_share"])} '
            f'unremarkable {fmt(s["unremarkable_share"])} contradicted {fmt(s["contradicted_share"])}  gap '
            f'{fmt_signed(gap)} -> {decide_gap(gap)}')


def table_lines(S, uncounted):
    """uncounted: {condition: {group: readings without claims}}"""
    L = []
    for c in CONDITIONS:
        L.append(f'== condition {c}, over readings with claims')
        L += [cell_line(g, S.get(c, {}).get(g)) for g in GROUPS + ('all',)]
        L.append('not counted (no reading, no explanation or no valid claims): '
                 + ', '.join(f'{g} {uncounted[c][g]}' for g in GROUPS))
    return L


def unit_section(name, e):
    L = [f'== {name}  ({e["group"]})', pct_line(e['pct'])]
    for rid in READINGS:
        r = e['readings'][rid]
        L.append(f'{rid}  ' + (claims_text(r['claims'], e['pct']) if r['claims'] is not None else r['status']))
    return '\n'.join(L)


def entry(no_text_status, text, c):
    """one reading's status and claims"""
    if text is None:
        return {'status': no_text_status, 'claims': None}
    if c is None:
        return {'status': 'claims call failed (not cached)', 'claims': None}
    if not c['valid']:
        return {'status': 'claims missing (no valid reply in 2 attempts)', 'claims': None}
    return {'status': 'claims', 'claims': c['claims']}


# ---------- selftest ----------

def selftest():
    import tempfile
    import threading
    import torch

    # constants
    assert READINGS == ['base_s1', 'base_s2', 'base_s3', 'P_s1', 'P_s2', 'P_s3'] and WORKERS == 6
    assert [cond_of(r) for r in READINGS] == ['base'] * 3 + ['P'] * 3
    assert PROPERTY_TRANSFORM == {'colour': 'grayscale', 'fine_detail': 'low_pass', 'coarse_shading': 'high_pass',
                                  'shape': 'phase_scramble', 'arrangement': 'patch_shuffle_4x4',
                                  'orientation': 'rotate_90', 'upright': 'rotate_180', 'left_right': 'mirror',
                                  'inner_texture': 'edge_map', 'polarity': 'luminance_polarity_inversion'}
    assert list(PROPERTY_TRANSFORM) == list(PROPERTIES)
    assert sorted(PROPERTY_TRANSFORM.values()) == sorted(battery.TRANSFORMS) and len(battery.TRANSFORMS) == 10
    assert {L: len(EXCLUDE[L]) for L in RANDOM_LAYERS} == {'mixed3a': 18, 'mixed3b': 13, 'mixed4b': 9,
                                                           'mixed4d': 10, 'mixed4e': 6, 'mixed5a': 1}
    assert all(0 <= c < NCH[L] for L in EXCLUDE for c in EXCLUDE[L])

    # unit draw: determinism, exclusions, no repeats, one generator for all layers in order
    nch = OrderedDict((L, NCH[L]) for L in RANDOM_LAYERS)
    a = draw_units(nch, EXCLUDE)
    assert a == draw_units(nch, EXCLUDE) == draw_units(nch, EXCLUDE, seed=22), 'draw not reproducible'
    assert [L for L, _ in a] == [L for L in RANDOM_LAYERS for _ in range(PER_LAYER)]
    assert len(set(a)) == len(a) == 18
    assert all(type(c) is int and 0 <= c < NCH[L] and c not in EXCLUDE[L] for L, c in a)
    rng, want = np.random.default_rng(22), []
    for L, n in nch.items():
        kept = []
        while len(kept) < 3:
            c = int(rng.integers(n))
            if c not in EXCLUDE[L] and c not in kept:
                kept.append(c)
        want += [(L, c) for c in kept]
    assert a == want, (a, want)
    assert draw_units(nch, EXCLUDE, seed=23) != a
    t = draw_units(OrderedDict([('x', 5)]), {'x': {0, 1}})
    assert [L for L, _ in t] == ['x'] * 3 and sorted(c for _, c in t) == [2, 3, 4]
    t = draw_units(OrderedDict([('x', 3), ('y', 4)]), {'y': {3}})
    assert [L for L, _ in t] == ['x'] * 3 + ['y'] * 3
    assert sorted(c for L, c in t if L == 'x') == [0, 1, 2] and sorted(c for L, c in t if L == 'y') == [0, 1, 2]
    for bad in [({'x': 4}, {'x': {0, 1}}), ({'x': 2}, {})]:
        try:
            draw_units(*bad)
        except ValueError:
            continue
        raise AssertionError(f'impossible draw accepted {bad}')

    # claim prompt
    lines = claims_prompt('EXPL').split('\n')
    assert len(lines) == 16 and lines[1] == lines[3] == '' and lines[2] == 'EXPL'
    assert lines[0] == ('Here is an explanation of what one unit (neuron) of an image classification network '
                        'responds to.')
    assert lines[4] == ('For each property below, say whether the explanation claims that the unit\'s response '
                        'depends on it ("depends"), claims that the response is unaffected by it ("unaffected"), or '
                        'does not say ("not_said"). Count only what the explanation states or clearly implies about '
                        'the unit\'s response, not descriptions of the photos.')
    assert [x[2:].split(':')[0] for x in lines[5:15]] == list(PROPERTIES) and all(x[:2] == '- ' for x in lines[5:15])
    assert lines[15] == 'Reply with only JSON with exactly these keys: {"colour": ..., "fine_detail": ..., ...}'
    assert '  ' not in claims_prompt('EXPL') and claims_prompt('a {text} b').count('a {text} b') == 1

    # claim parsing: good, missing key, extra key, bad value
    none_obj = {p: 'not_said' for p in PROPERTIES}
    good_obj = dict(none_obj, colour='depends', shape='depends', polarity='unaffected')
    good = json.dumps(good_obj)
    assert parse_claims(good) == good_obj and parse_claims(json.dumps(none_obj)) == none_obj
    assert parse_claims('Here:\n```json\n' + json.dumps(good_obj, indent=2) + '\n```\nDone.') == good_obj
    miss = dict(good_obj)
    del miss['polarity']
    assert parse_claims(json.dumps(miss)) is None
    assert parse_claims(json.dumps(dict(good_obj, texture='depends'))) is None
    for bad in ('yes', 'Depends', 'depends ', '', None, True, 1, ['depends'], {'v': 'depends'}):
        assert parse_claims(json.dumps(dict(good_obj, colour=bad))) is None, bad
    assert parse_claims('no json here') is None and parse_claims('') is None and parse_claims(None) is None
    assert parse_claims(good + ' or rather ' + json.dumps(dict(good_obj, colour='maybe'))) is None
    assert parse_claims(json.dumps(miss) + ' corrected: ' + good) == good_obj

    # verdicts at the boundaries 20 and 80
    for p, d, u in [(0, 'supported', 'contradicted'), (20, 'supported', 'contradicted'),
                    (20.0001, 'unremarkable', 'unremarkable'), (50, 'unremarkable', 'unremarkable'),
                    (79.9999, 'unremarkable', 'unremarkable'), (80, 'contradicted', 'supported'),
                    (100, 'contradicted', 'supported'), (float('nan'), 'unremarkable', 'unremarkable')]:
        assert verdict('depends', p) == d and verdict('unaffected', p) == u, (p, d, u)
    assert verdict('not_said', 20) is None and verdict('not_said', 80) is None and verdict('not_said', 50) is None
    for bad in ('maybe', 'Depends', None):
        try:
            verdict(bad, 50)
        except ValueError:
            continue
        raise AssertionError(f'bad claim accepted {bad!r}')

    # summarize on a hand case
    def cl(**kw):
        return dict(none_obj, **kw)
    pA = dict({t: 50.0 for t in battery.TRANSFORMS}, grayscale=10.0, low_pass=90.0, phase_scramble=20.0,
              patch_shuffle_4x4=80.0)
    pB = dict({t: 50.0 for t in battery.TRANSFORMS}, phase_scramble=85.0)
    recs = [
        {'group': 'random', 'condition': 'base', 'pct': pA,  # S, C, U
         'claims': cl(colour='depends', fine_detail='depends', coarse_shading='unaffected')},
        {'group': 'random', 'condition': 'base', 'pct': pA, 'claims': cl()},  # no claims
        {'group': 'round21', 'condition': 'base', 'pct': pB, 'claims': cl(shape='unaffected', arrangement='depends')},
        {'group': 'random', 'condition': 'P', 'pct': pA,  # S, S, S, C
         'claims': cl(fine_detail='unaffected', shape='depends', arrangement='unaffected', colour='unaffected')},
        {'group': 'round21', 'condition': 'P', 'pct': pB, 'claims': None},  # missing: not a reading with claims
    ]
    S = summarize(recs)

    def cell(n, k, s, u, c):
        return {'readings': n, 'claims_per_reading': k / n, 'supported_per_reading': s / n,
                'unremarkable_per_reading': u / n, 'contradicted_per_reading': c / n,
                'supported_share': s / k if k else None, 'unremarkable_share': u / k if k else None,
                'contradicted_share': c / k if k else None}

    def same(x, y):
        assert list(x) == list(y), (list(x), list(y))
        for k in y:
            assert (x[k] is None and y[k] is None) or (y[k] is not None and abs(x[k] - y[k]) < 1e-12), (k, x, y)
    assert list(S) == ['base', 'P'] and list(S['base']) == ['random', 'round21', 'all']
    assert list(S['P']) == ['random', 'all']
    same(S['base']['random'], cell(2, 3, 1, 1, 1))
    same(S['base']['round21'], cell(1, 2, 1, 1, 0))
    same(S['base']['all'], cell(3, 5, 2, 2, 1))
    same(S['P']['random'], cell(1, 4, 3, 0, 1))
    same(S['P']['all'], cell(1, 4, 3, 0, 1))
    assert abs(S['base']['all']['supported_share'] - 0.4) < 1e-12 and S['P']['random']['contradicted_share'] == 0.25
    z = summarize([recs[1]])
    assert list(z) == ['base'] and list(z['base']) == ['random', 'all']
    same(z['base']['all'], cell(1, 0, 0, 0, 0))
    assert z['base']['random']['supported_share'] is None and z['base']['random']['claims_per_reading'] == 0
    assert summarize([]) == {} and summarize([recs[4]]) == {}
    part = summarize([{'group': 'g', 'condition': 'c', 'claims': {'upright': 'depends'}, 'pct': {'rotate_180': 5}}])
    same(part['c']['g'], cell(1, 1, 1, 0, 0))
    for bad in ([{'group': 'random', 'condition': 'base', 'claims': {'colour': 'maybe'}, 'pct': pA}],
                [{'group': 'all', 'condition': 'base', 'claims': cl(), 'pct': pA}]):
        try:
            summarize(bad)
        except ValueError:
            continue
        raise AssertionError('bad records accepted')
    json.dumps(S)

    # neutral labels, mid-profile filter, report lines
    assert share_gap(S['P']['random']) == 0.5 and share_gap(z['base']['all']) is None and share_gap(None) is None
    assert decide_gap(0.3) == 'SUPPORTED_MORE' and decide_gap(-0.3) == 'CONTRADICTED_MORE'
    assert decide_gap(0.25) == decide_gap(-0.25) == decide_gap(0.0) == 'SIMILAR'
    assert decide_gap(None) == decide_gap(float('nan')) == 'UNDEFINED'
    mid = {t: 50.0 for t in battery.TRANSFORMS}
    assert mid_profile(mid) and mid_profile(dict(mid, mirror=20.5)) and mid_profile(dict(mid, mirror=79.5))
    for p in (20, 80, 0, 100, float('nan')):
        assert not mid_profile(dict(mid, edge_map=p)), p
    pc = dict(mid, grayscale=10.0, low_pass=80.0, mirror=float('nan'))
    c4 = cl(colour='depends', fine_detail='depends', left_right='unaffected', shape='unaffected')
    assert claims_text(c4, pc) == ('colour:depends:supported fine_detail:depends:contradicted '
                                   'shape:unaffected:unremarkable left_right:unaffected:unremarkable')
    assert claims_text(cl(), pc) == '(no claims)'
    pl = pct_line(pc)
    assert pl.startswith('percentiles: grayscale 10.0, low_pass 80.0, high_pass 50.0, ') and 'mirror n/a' in pl
    assert cell_line('round21', None) == 'round21  no readings with claims'
    ln = cell_line('random', S['P']['random'])
    assert ln.startswith('random   readings 1  claims/reading 4.000  per reading: supported 3.000 '), ln
    assert ln.endswith('gap +0.500 -> SUPPORTED_MORE'), ln
    assert cell_line('all', z['base']['all']).endswith('gap n/a -> UNDEFINED')
    tl = table_lines(S, {c: {g: 0 for g in GROUPS} for c in CONDITIONS})
    assert len(tl) == 2 * 5 and tl[0] == '== condition base, over readings with claims'
    assert tl[5] == '== condition P, over readings with claims' and tl[6].startswith('random   readings 1')
    assert tl[7] == 'round21  no readings with claims' and tl[9].endswith(': random 0, round21 0')
    e = {'group': 'random', 'pct': pc,
         'readings': {rid: {'status': f'st {rid}', 'claims': None} for rid in READINGS}}
    e['readings']['P_s2'] = {'status': 'claims', 'claims': c4}
    sec = unit_section('mixed3a:5', e).splitlines()
    assert sec[0] == '== mixed3a:5  (random)' and sec[1] == pl and len(sec) == 2 + len(READINGS)
    assert sec[2] == 'base_s1  st base_s1' and sec[6] == 'P_s2  ' + claims_text(c4, pc)
    assert entry('none', None, None)['status'] == 'none' and entry('x', 't', None)['claims'] is None
    assert entry('x', 't', {'valid': False, 'claims': None})['status'].startswith('claims missing')
    assert entry('x', 't', {'valid': True, 'claims': c4}) == {'status': 'claims', 'claims': c4}
    assert clean(float('nan')) is None and clean(None) is None and clean(np.float32(2.5)) == 2.5

    # claim calls: one retry in a fresh call, then missing; an is_error reply raises
    lock, calls = threading.Lock(), []

    def scripted(outs):
        def call(content):
            assert len(content) == 1 and content[0]['type'] == 'text'
            with lock:
                calls.append(content[0]['text'])
                return outs.pop(0)
        return call
    g = get_claims('some text', scripted([{'text': 'nope', 'is_error': False}, {'text': good, 'is_error': False}]))
    assert g['valid'] and g['claims'] == good_obj and g['attempts'] == 2 and g['replies'] == ['nope', good]
    assert g['reply'] == good and calls == [claims_prompt('some text')] * 2
    calls.clear()
    g = get_claims('t', scripted([{'text': good, 'is_error': False}]))
    assert g['valid'] and g['attempts'] == 1 and len(calls) == 1
    calls.clear()
    g = get_claims('t', scripted([{'text': json.dumps(miss), 'is_error': False}, {'text': 'x', 'is_error': False},
                                  {'text': good, 'is_error': False}]))
    assert not g['valid'] and g['claims'] is None and g['attempts'] == 2 and len(calls) == 2
    for outs in ([{'text': 'overloaded', 'is_error': True}],
                 [{'text': 'x', 'is_error': False}, {'text': 'overloaded', 'is_error': True}]):
        try:
            get_claims('t', scripted(outs))
        except RuntimeError:
            continue
        raise AssertionError('is_error reply accepted')

    # claims cache: keyed by reading id with the text hash; is_error not cached; a changed text stops
    def claimer(n_err):
        state = {'err': n_err}

        def call(content):
            t = content[0]['text']
            with lock:
                calls.append(t)
                if 'ERR' in t and state['err']:
                    state['err'] -= 1
                    return {'text': 'overloaded', 'is_error': True}
            return {'text': good if 'GOOD' in t else 'no json', 'is_error': False}
        return call
    texts = {'base_s1': 'GOOD one', 'P_s2': 'BAD two', 'base_s3': 'ERR three'}
    with tempfile.TemporaryDirectory() as td:
        calls.clear()
        c1 = unit_claims('mixed3a:5', texts, Path(td), call=claimer(1))
        assert sorted(c1) == ['P_s2', 'base_s1'] and len(calls) == 4
        assert c1['base_s1']['valid'] and c1['base_s1']['claims'] == good_obj and c1['base_s1']['condition'] == 'base'
        assert c1['base_s1']['text_sha256'] == openmethod.sha256('GOOD one') and c1['base_s1']['reading'] == 'base_s1'
        assert not c1['P_s2']['valid'] and c1['P_s2']['claims'] is None and c1['P_s2']['attempts'] == 2
        calls.clear()
        c2 = unit_claims('mixed3a:5', texts, Path(td), call=claimer(0))
        assert sorted(c2) == sorted(texts) and len(calls) == 2 and not c2['base_s3']['valid']
        calls.clear()
        assert unit_claims('mixed3a:5', texts, Path(td), call=claimer(0)) == c2 and calls == []
        try:
            unit_claims('mixed3a:5', dict(texts, base_s1='GOOD changed'), Path(td), call=claimer(0))
            raise AssertionError('changed explanation accepted')
        except SystemExit:
            pass
        assert calls == []

    # base and P contents exactly as lookviews builds them; readings cache
    cb, rb = {'type': 'image', 'k': 'card'}, {'type': 'image', 'k': 'random'}
    nm = bundlemax.card_name('mixed4c:447', 'whole')
    cc = openmethod.card_content(nm, cb, rb)
    cs = reading_contents(cc, 'PROFILE')
    full = lookviews.condition_contents(cc[0]['text'], cb, rb, {'type': 'image', 'k': 'crops'}, 'PROFILE', 48)
    assert list(cs) == ['base', 'P'] and cs['base'] == full['base'] == cc and cs['P'] == full['P']
    T = run_subjects.TASK
    assert [b.get('k', b.get('text')) for b in cs['P']] == [cc[0]['text'], 'card', 'random', 'PROFILE', T]

    def reader(n_err):
        state = {'err': n_err}

        def call(sp, content):
            assert sp == ''
            with lock:
                calls.append(len(content))
                if state['err'] and len(content) == 5:
                    state['err'] -= 1
                    return {'text': 'overloaded', 'is_error': True}
            return {'text': '{"class": "one", "explanation": "n=%d"}' % len(content), 'is_error': False}
        return call
    with tempfile.TemporaryDirectory() as td:
        calls.clear()
        rec = readings('mixed4c:447', cs, Path(td), call=reader(1))
        assert len(rec) == 5 and len(calls) == 6 and sum(k.startswith('P_') for k in rec) == 2
        assert rec['base_s2']['explanation'] == 'n=4' and rec['base_s1']['condition'] == 'base'
        assert rec['base_s3']['sample'] == 3 and rec['base_s1']['system_prompt'] == ''
        calls.clear()
        rec = readings('mixed4c:447', cs, Path(td), call=reader(0))
        assert sorted(rec) == sorted(READINGS) and calls == [5] and rec['P_s1']['explanation'] == 'n=5'
        calls.clear()
        cs2 = dict(cs, P=cs['P'][:3] + [{'type': 'text', 'text': 'OTHER'}] + cs['P'][4:])
        try:
            readings('mixed4c:447', cs2, Path(td), call=reader(0))
            raise AssertionError('changed content accepted')
        except SystemExit:
            pass
        assert calls == []

    # values: openmethod's definition and values.npz format, on small stand-in modules (not the network)
    class Boom(torch.nn.Module):
        def forward(self, x):
            raise AssertionError('ran past the stop')

    def stand_in(seed):
        torch.manual_seed(seed)
        m = torch.nn.Sequential(OrderedDict([('mixed4c', torch.nn.Conv2d(3, 512, 16, stride=16)), ('boom', Boom())]))
        for q in m.parameters():
            q.requires_grad_(False)
        return m.eval()
    fm, fm2 = stand_in(0), stand_in(1)
    u = openmethod.parse_unit('mixed4c:5')
    ids = np.arange(5, dtype=np.int64)
    got, ref = forward_max_on(fm, u, ids, 'cpu'), openmethod.forward_max(fm, u, ids)
    assert all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in zip(got[:3], ref[:3]))
    assert got[3] == ref[3] == 14
    if torch.cuda.is_available():
        import copy
        exact_cuda()
        on_gpu = forward_max_on(copy.deepcopy(fm).to('cuda'), u, ids, 'cuda')
        assert np.allclose(on_gpu[0], ref[0], rtol=1e-3) and on_gpu[3] == 14
    with tempfile.TemporaryDirectory() as td:
        pa, pb, pc_ = Path(td) / 'a', Path(td) / 'b', Path(td) / 'c'
        for p in (pa, pb, pc_):
            p.mkdir()
        Cs, Hs = np.array([0, 2, 4], dtype=np.int64), np.array([1, 3], dtype=np.int64)
        d1 = unit_values(u, Cs, Hs, pa, lambda: fm, lambda: fm, 'cpu')
        openmethod.unit_values(u, Cs, Hs, pb, lambda: fm)
        with np.load(pa / 'values.npz') as za, np.load(pb / 'values.npz') as zb:
            assert sorted(za.files) == sorted(zb.files)
            for k in za.files:
                assert za[k].dtype == zb[k].dtype and np.array_equal(za[k], zb[k]), k
        d3 = openmethod.unit_values(u, Cs, Hs, pa, None)  # openmethod's cache check accepts the file
        assert np.array_equal(d3['vC'], d1['vC']) and np.array_equal(d3['cH'], d1['cH'])
        d4 = unit_values(u, Cs, Hs, pa, None, None, 'cpu')  # cached: no model needed
        assert np.array_equal(d4['vH'], d1['vH'])
        try:
            unit_values(u, Cs, Hs, pc_, lambda: fm, lambda: fm2, 'cpu')
            raise AssertionError('values differing from the CPU computation accepted')
        except SystemExit:
            pass
        assert not (pc_ / 'values.npz').exists()
    print('selftest ok')


# ---------- main ----------

def run():
    import torch
    reply = json.loads(lookviews.REPLY.read_text())
    desc = {n: lookviews.describe(n, reply) for n in battery.TRANSFORMS}
    names = json.loads(openmethod.nearmiss2.NAMES.read_text())
    C, H = bundlemax.pools(names)
    P0 = np.sort(np.random.default_rng(0).choice(C, lookviews.N_PROFILE, replace=False))

    # 1. units
    rall = json.loads((LV / 'readings_all.json').read_text())
    r21 = [openmethod.parse_unit(n) for n in rall]
    if len(r21) != N_ROUND21:
        raise SystemExit(f'{LV / "readings_all.json"}: {len(r21)} units, expected {N_ROUND21}; stopping')
    rnd = [openmethod.parse_unit(f'{L}:{c}')
           for L, c in draw_units(OrderedDict((L, NCH[L]) for L in RANDOM_LAYERS), EXCLUDE, SEED)]
    both = sorted({u['name'] for u in rnd} & {u['name'] for u in r21})
    if both:
        raise SystemExit(f'drawn units {both} are also round-21 units (their claims would share a cache); stopping')
    units = [(u, 'random') for u in rnd] + [(u, 'round21') for u in r21]
    print('random units:', ', '.join(u['name'] for u in rnd), flush=True)
    print('round-21 units:', ', '.join(u['name'] for u in r21), flush=True)

    # 2. profiles, from the runs/lookviews caches only
    layers = sorted({u['layer'] for u, _ in units}, key=lookviews.LAYERS.index)
    missing = [L for L in layers if not (LV / f'profile_{L}.npz').exists()]
    if missing:
        raise SystemExit(f'no cached profile in {LV} for {missing}; stopping')

    def no_model():
        raise SystemExit('a profile is not cached; stopping')
    prof = lookviews.profiles(layers, P0, no_model)
    R = {L: lookviews.retention_table(prof[L]) for L in layers}
    info, prows = {}, {}
    for u, g in units:
        cnt, rows = lookviews.unit_profile(R[u['layer']], u['chan'])
        prows[u['name']] = (cnt, rows)
        info[u['name']] = {'group': g, 'layer': u['layer'], 'chan': u['chan'], 'layer_channels': cnt,
                           'pct': {t: p for t, _, _, p in rows}, 'readings': {}}

    # 3. random units: values (CUDA when available), checked against the profile originals as lookviews does
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        exact_cuda()
    print('values on', dev, flush=True)
    held = {}

    def get_model(d):
        if d not in held:
            held[d] = load_net(d)
        return held[d]
    vals = {}
    for u in rnd:
        n = u['name']
        print('==', n, flush=True)
        udir = OUT / n
        udir.mkdir(parents=True, exist_ok=True)
        d = unit_values(u, C, H, udir, lambda: get_model(dev), lambda: get_model('cpu'), dev)
        pos = np.searchsorted(d['C'], P0)
        if not np.array_equal(d['C'][pos], P0):
            raise SystemExit(f'{n}: P0 is not inside C; stopping')
        M = float(d['vC'].max())
        if not np.allclose(prof[u['layer']]['orig'][:, u['chan']], d['vC'][pos], rtol=1e-3, atol=1e-3 * M):
            raise SystemExit(f'{n}: profile originals differ from values.npz; stopping')
        vals[n] = d
    del prof
    held.clear()
    gc.collect()
    if dev == 'cuda':
        torch.cuda.empty_cache()

    # 4. random units: card, base and P contents, readings
    recs = {}
    for u in rnd:
        n = u['name']
        udir = OUT / n
        _meta, content = openmethod.make_cards(u, vals[n], udir)
        assert [b['type'] for b in content] == ['text', 'image', 'image', 'text']
        assert content[3]['text'] == run_subjects.TASK
        cnt, rows = prows[n]
        contents = reading_contents(content, lookviews.profile_text(cnt, rows, desc))
        recs[n] = readings(n, contents, udir)
        del content, contents
        gc.collect()
    del vals

    # 5. claims of every base and P reading
    for u, g in units:
        n = u['name']
        udir = OUT / n
        udir.mkdir(parents=True, exist_ok=True)
        if g == 'random':
            src = {rid: (recs[n][rid]['explanation'] if rid in recs[n] else None) for rid in READINGS}
            why = {rid: ('no explanation (reading reply not parsed)' if rid in recs[n]
                         else 'no reading (call failed, not cached)') for rid in READINGS}
        else:
            src = {rid: rall[n].get(rid) for rid in READINGS}
            why = {rid: 'no explanation in runs/lookviews/readings_all.json' for rid in READINGS}
        texts = {rid: t for rid, t in src.items() if t is not None}
        cl = unit_claims(n, texts, udir)
        info[n]['readings'] = {rid: entry(why[rid], src[rid], cl.get(rid)) for rid in READINGS}

    # 6. summary and report
    records = [{'unit': n, 'group': e['group'], 'condition': cond_of(rid), 'claims': r['claims'], 'pct': e['pct']}
               for n, e in info.items() for rid, r in e['readings'].items() if r['claims'] is not None]
    S = summarize(records)
    uncounted = {c: {g: sum(1 for e in info.values() if e['group'] == g for rid in READINGS
                            if cond_of(rid) == c and e['readings'][rid]['claims'] is None) for g in GROUPS}
                 for c in CONDITIONS}
    mids = [n for n, e in info.items() if e['group'] == 'random' and mid_profile(e['pct'])]
    Sm = summarize([r for r in records if r['unit'] in mids])
    mid_by = {}
    for c in CONDITIONS:
        s = Sm.get(c, {}).get('random')
        mid_by[c] = {'readings': 0 if s is None else s['readings'],
                     'claims_per_reading': None if s is None else s['claims_per_reading']}

    head = [
        'profilecheck: claims in base and P explanations about 10 properties, against the invariance profile',
        'verdict on p = the unit\'s percentile for the matching transform (lookviews.unit_profile): "depends" '
        f'supported if p <= {P_LOW}, contradicted if p >= {P_HIGH}; "unaffected" supported if p >= {P_HIGH}, '
        f'contradicted if p <= {P_LOW}; otherwise unremarkable',
        'property -> transform: ' + ', '.join(f'{p} {t}' for p, t in PROPERTY_TRANSFORM.items()),
        f'random units (default_rng({SEED}), {PER_LAYER} per layer): ' + ', '.join(u['name'] for u in rnd),
        'round-21 units (runs/lookviews/readings_all.json): ' + ', '.join(u['name'] for u in r21),
        f'label per condition and group on gap = supported share - contradicted share: > {GAP} SUPPORTED_MORE, '
        f'< -{GAP} CONTRADICTED_MORE, else SIMILAR; no claims UNDEFINED',
    ]
    mid_lines = [f'== random units with all 10 percentiles in ({P_LOW}, {P_HIGH}): {len(mids)}'
                 + (': ' + ', '.join(mids) if mids else '')]
    mid_lines += [f'{c:<4} readings {mid_by[c]["readings"]}  claims/reading {fmt(mid_by[c]["claims_per_reading"])}'
                  for c in CONDITIONS]
    secs = [unit_section(n, e) for n, e in info.items()]
    text = ('\n'.join(head) + '\n\n' + '\n'.join(table_lines(S, uncounted)) + '\n\n' + '\n'.join(mid_lines)
            + '\n\n' + '\n\n'.join(secs) + '\n')
    (OUT / 'report.txt').write_text(text)

    summary = {
        'random_units': [u['name'] for u in rnd], 'round21_units': [u['name'] for u in r21], 'seed': SEED,
        'thresholds': {'p_low': P_LOW, 'p_high': P_HIGH, 'gap': GAP}, 'property_transform': PROPERTY_TRANSFORM,
        'summary': S,
        'labels': {c: {g: {'gap': share_gap(s), 'label': decide_gap(share_gap(s))} for g, s in S[c].items()}
                   for c in S},
        'not_counted': uncounted,
        'mid_profile_random': {'units': mids, 'by_condition': mid_by},
        'units': {n: {'group': e['group'], 'layer': e['layer'], 'chan': e['chan'],
                      'layer_channels': e['layer_channels'],
                      'percentiles': {t: clean(p) for t, p in e['pct'].items()},
                      'readings': {rid: {'status': r['status'], 'claims': r['claims'],
                                         'verdicts': None if r['claims'] is None else
                                         {p: [v, vd] for p, v, vd in claim_items(r['claims'], e['pct'])}}
                                   for rid, r in e['readings'].items()}}
                  for n, e in info.items()},
    }
    lookviews.save_json(summary, OUT / 'summary.json')
    print(text, flush=True)
    print('report ->', OUT / 'report.txt')


def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.parse_args()
    run()


if __name__ == '__main__':
    main()
