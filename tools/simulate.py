"""Simulation scores for image units (spec: authors/spec_simulate.md).

An agent that sees only an explanation predicts the unit's activation, from 0 to 10, on every photo of a fixed pool of
48; the Spearman rank correlation between predicted and recorded activation scores the explanation.

    python -I tools/simulate.py --units mixed4c:447,mixed4e:55 [--selftest]
"""
import argparse
import base64
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I does not add the script dir
import claude_call  # noqa: E402
from nearmiss import FIRST  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPL = os.path.join(ROOT, 'authors/explanations/nearmiss2.json')
SRC = os.path.join(ROOT, 'runs/nearmiss2')
OUT = os.path.join(ROOT, 'runs/simulate')
SAMPLES, N, TOP, PMAX = 3, 48, 12, 10
KEYS = {str(k) for k in range(1, N + 1)}
CHANCE = {'rho': 0.0, 'top12': 0.25}

LAST = ("Using only the explanation, predict the unit's activation on every photo, from 0 (silent) to 10 "
        '(as strong as on the photos it responds to most). Reply with only JSON: '
        '{"predictions": {"1": n, "2": n, ..., "48": n}}')


def pool_ratios(pool):
    """the recorded ratios ordered by photo number; only "number" and "ratio" are read"""
    if not isinstance(pool, list) or len(pool) != N:
        raise ValueError(f'pool must be a list of {N} records')
    if sorted(p['number'] for p in pool) != list(range(1, N + 1)):
        raise ValueError(f'pool numbers must be 1-{N}')
    by_number = {p['number']: float(p['ratio']) for p in pool}
    ratio = [by_number[k] for k in range(1, N + 1)]
    if not all(math.isfinite(r) for r in ratio):
        raise ValueError('pool has a non-finite ratio')
    return ratio


def parse_reply(text):
    """the reply's JSON object as claude_call.parse_json finds it, or None"""
    try:
        return claude_call.parse_json(text)
    except ValueError:
        return None


def predictions(obj):
    """the 48 predictions ordered by photo number, or None if obj is not a valid reply"""
    if not isinstance(obj, dict) or not isinstance(obj.get('predictions'), dict):
        return None
    got = {}
    for k, v in obj['predictions'].items():
        if type(k) is int and 1 <= k <= N:
            num = k
        elif type(k) is str and k in KEYS:
            num = int(k)
        else:
            return None
        if num in got or type(v) not in (int, float) or not 0 <= v <= PMAX:
            return None
        got[num] = float(v)
    return [got[k] for k in range(1, N + 1)] if len(got) == N else None


def run_sample(text, sheet_b64):
    content = [{'type': 'text', 'text': FIRST.format(text=text)},
               {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': sheet_b64}},
               {'type': 'text', 'text': LAST}]
    replies = []
    for _ in range(2):  # one retry, fresh conversation
        out = claude_call.converse('', [content])[0]
        if out['is_error']:  # raised, so the sample is not cached and a rerun tries again
            raise RuntimeError('is_error reply: ' + str(out['text'])[:200])
        replies.append(out['text'])
        pred = predictions(parse_reply(out['text']))
        if pred is not None:
            return {'replies': replies, 'valid': True, 'predictions': pred}
    return {'replies': replies, 'valid': False}


def ranks(x):
    """1-based ranks, tied values given their average rank"""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='stable')
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    """Pearson correlation of the average ranks; 0 if either side is constant"""
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def top12(pred, ratio):
    """share of the 12 highest-ratio photos among the 12 highest predictions; ties broken by photo number"""
    def top(v):
        return set(sorted(range(len(v)), key=lambda i: (-v[i], i))[:TOP])
    return len(top(pred) & top(ratio)) / TOP


def score_sample(pred, ratio):
    const = len(set(pred)) == 1
    return {'rho': 0.0 if const else spearman(pred, ratio), 'top12': top12(pred, ratio), 'constant': const,
            'predictions': list(pred)}


def decision(rhos):
    """rho against its chance level 0, over valid samples"""
    if not rhos:
        return 'NONE'
    return 'ABOVE' if min(rhos) > 0 else 'BELOW' if max(rhos) < 0 else 'MIXED'


def summarize(recs, ratio):
    """scores of one explanation from its cached records {sample: record}, over the valid samples"""
    valid = {s: r['predictions'] for s, r in sorted(recs.items(), key=lambda kv: int(kv[0])) if r['valid']}
    samples = {s: score_sample(p, ratio) for s, p in valid.items()}
    rhos = [v['rho'] for v in samples.values()]
    preds = list(valid.values())
    out = {'n_cached': len(recs), 'n_valid': len(valid), 'samples': samples,
           'constant': [s for s, v in samples.items() if v['constant']], 'rho_vs_chance': decision(rhos)}
    if not valid:
        return {**out, 'rho_mean': None, 'rho_min': None, 'rho_max': None, 'mean_prediction': None,
                'rho_mean_prediction': None, 'consistency': None, 'top12_mean': None}
    mean_pred = [math.fsum(col) / len(preds) for col in zip(*preds)]  # fsum: equal means stay exactly tied
    pairs = [spearman(a, b) for i, a in enumerate(preds) for b in preds[i + 1:]]
    return {**out, 'rho_mean': float(np.mean(rhos)), 'rho_min': min(rhos), 'rho_max': max(rhos),
            'mean_prediction': mean_pred, 'rho_mean_prediction': spearman(mean_pred, ratio),
            'consistency': float(np.mean(pairs)) if pairs else None,
            'top12_mean': float(np.mean([v['top12'] for v in samples.values()]))}


def fmt(x):
    return 'n/a' if x is None else f'{x:.3f}'


def unit_report(unit, expl, summ):
    lines = [f'== {unit}  photos={N}  chance: rho={CHANCE["rho"]:.3f}  top12={CHANCE["top12"]:.3f}']
    for n, e in expl.items():
        s = summ[n]
        const = 'none' if not s['constant'] else 'samples ' + ','.join(s['constant'])
        lines.append(f'{n}  source={e["source"]}  rho={fmt(s["rho_mean"])} ({fmt(s["rho_min"])} to {fmt(s["rho_max"])})'
                     f'  rho_mean_pred={fmt(s["rho_mean_prediction"])}  consistency={fmt(s["consistency"])}'
                     f'  top12={fmt(s["top12_mean"])}  valid={s["n_valid"]}/{SAMPLES}  constant={const}'
                     f'  rho_vs_chance={s["rho_vs_chance"]}')
    ranked = sorted((n for n in expl if summ[n]['n_valid']), key=lambda n: -summ[n]['rho_mean'])
    none = [n for n in expl if not summ[n]['n_valid']]
    lines.append('ranked by mean rho: ' + (', '.join(f'{n} ({summ[n]["rho_mean"]:.3f})' for n in ranked) or 'none'))
    if none:
        lines.append('no valid samples: ' + ', '.join(none))
    return lines, ranked, none


def save_json(obj, path):
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(path + '.tmp', path)


def main(units):
    expl = json.load(open(EXPL))
    unknown = [u for u in units if u not in expl]
    if unknown:
        raise ValueError(f'units not in {EXPL}: {unknown}')
    ratios, sheets, caches, cache_paths = {}, {}, {}, {}
    for u in units:
        ratios[u] = pool_ratios(json.load(open(os.path.join(SRC, u, 'pool.json'))))
        sheets[u] = base64.b64encode(open(os.path.join(SRC, u, 'sheet.jpg'), 'rb').read()).decode()
        os.makedirs(os.path.join(OUT, u), exist_ok=True)
        cache_paths[u] = os.path.join(OUT, u, 'replies.json')
        caches[u] = json.load(open(cache_paths[u])) if os.path.exists(cache_paths[u]) else {}

    jobs = [(u, n, s) for u in units for n in expl[u] for s in range(SAMPLES) if str(s) not in caches[u].get(n, {})]
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(run_sample, expl[u][n]['text'], sheets[u]): (u, n, s) for u, n, s in jobs}
        for f in as_completed(futs):
            u, n, s = futs[f]
            try:
                rec = f.result()
            except Exception as e:  # not cached, so a rerun tries again
                print(f'{u} {n} sample {s} failed: {e}')
                continue
            caches[u].setdefault(n, {})[str(s)] = rec
            save_json(caches[u], cache_paths[u])

    blocks = []
    for u in units:
        summ = {n: summarize(caches[u].get(n, {}), ratios[u]) for n in expl[u]}
        lines, ranked, none = unit_report(u, expl[u], summ)
        blocks.append('\n'.join(lines))
        save_json({'unit': u, 'n_photos': N, 'samples_per_explanation': SAMPLES, 'chance': CHANCE,
                   'explanations': {n: {'source': expl[u][n]['source'], **summ[n]} for n in expl[u]},
                   'ranked_by_mean_rho': ranked, 'no_valid_samples': none}, os.path.join(OUT, u, 'scores.json'))
    report = '\n\n'.join(blocks) + '\n'
    with open(os.path.join(OUT, 'report.txt'), 'w') as f:
        f.write(report)
    print(report, end='')


def selftest():
    # pool check
    pool = [{'number': k, 'ratio': k / 10, 'index': 99 - k} for k in range(N, 0, -1)]
    assert pool_ratios(pool) == [k / 10 for k in range(1, N + 1)]
    for bad in (pool[:-1], pool[:-1] + [dict(pool[0])], pool + [{'number': 49, 'ratio': 0.0}]):
        try:
            pool_ratios(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('bad pool accepted')

    # reply validation
    vals = {k: k % 11 for k in range(1, N + 1)}  # ints including 0 and 10
    vals[5] = 2.5
    good = {str(k): vals[k] for k in range(N, 0, -1)}  # key order should not matter

    def wrap(d):
        return 'Predictions:\n```json\n' + json.dumps({'predictions': d}) + '\n```\n{"note": 1}'

    p = predictions(parse_reply(wrap(good)))
    assert p == [float(vals[k]) for k in range(1, N + 1)]
    missing = {k: v for k, v in good.items() if k != '48'}
    padded = {**{k: v for k, v in good.items() if k != '7'}, '07': 7}
    bad = [missing, {**good, '49': 3}, {**good, '0': 3}, {**good, 'note': 3}, padded,
           {**good, '7': 10.5}, {**good, '7': -1}, {**good, '7': True}, {**good, '7': False},
           {**good, '7': '5'}, {**good, '7': None}, {**good, '7': [5]}]
    assert not any(predictions(parse_reply(wrap(d))) is not None for d in bad)
    assert predictions(parse_reply('no json')) is None
    assert predictions(parse_reply(json.dumps(good))) is None  # no "predictions" key
    assert predictions(parse_reply('{"predictions": [1, 2, 3]}')) is None
    ints = {k: vals[k] for k in range(1, N + 1)}
    assert predictions({'predictions': ints}) == p  # integer keys
    assert predictions({'predictions': {**{k: vals[k] for k in range(1, 25)},
                                        **{str(k): vals[k] for k in range(25, N + 1)}}}) == p
    assert predictions({'predictions': {**ints, '1': vals[1]}}) is None  # 1 and "1" both present
    assert predictions({'predictions': {**{k: vals[k] for k in range(2, N + 1)}, 49: 1}}) is None

    # Spearman against hand values, ties, constant
    assert np.allclose(ranks([10, 20, 20, 20, 5]), [2, 4, 4, 4, 1])
    assert np.isclose(spearman([1, 2, 2, 3], [1, 3, 2, 4]), math.sqrt(0.9))
    assert np.isclose(spearman([1, 1, 2, 2], [1, 2, 3, 4]), 2 / math.sqrt(5))
    assert np.isclose(spearman([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]), 1)
    assert np.isclose(spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1)
    assert spearman([3, 3, 3, 3], [1, 2, 3, 4]) == 0
    ratio = [float(N + 1 - k) for k in range(1, N + 1)]  # photo 1 highest
    s = score_sample([5.0] * N, ratio)
    assert s['rho'] == 0 and s['constant'] and s['top12'] == 1  # constant: top 12 by photo number = photos 1-12
    s = score_sample(ratio[::-1], ratio)
    assert np.isclose(s['rho'], -1) and s['top12'] == 0 and not s['constant']

    # top12 with ties
    assert top12([10.0 if 7 <= k <= 30 else 0.0 for k in range(1, N + 1)], ratio) == 0.5  # photos 7-18
    assert top12([10.0 if 13 <= k <= 24 else 0.0 for k in range(1, N + 1)], ratio) == 0
    tied = [1.0] * 24 + [0.0] * 24  # tied ratios: photos 1-12 by photo number
    assert top12(ratio, tied) == 1 and top12(ratio[::-1], tied) == 0

    # consistency and aggregation on synthetic samples
    a = [float(k) for k in range(N)]
    recs = {'0': {'valid': True, 'predictions': a}, '1': {'valid': True, 'predictions': list(a)},
            '2': {'valid': True, 'predictions': a[::-1]}}
    s = summarize(recs, ratio)
    assert np.isclose(s['consistency'], -1 / 3) and np.isclose(s['rho_mean'], -1 / 3)
    assert np.isclose(s['rho_min'], -1) and np.isclose(s['rho_max'], 1) and np.isclose(s['rho_mean_prediction'], -1)
    assert s['n_valid'] == 3 and s['constant'] == [] and s['rho_vs_chance'] == 'MIXED'
    recs = {'2': {'valid': False, 'replies': ['x', 'y']}, '0': {'valid': True, 'predictions': [5.0] * N},
            '1': {'valid': True, 'predictions': list(ratio)}}
    s = summarize(recs, ratio)
    assert s['n_valid'] == 2 and s['n_cached'] == 3 and s['constant'] == ['0'] and list(s['samples']) == ['0', '1']
    assert np.isclose(s['rho_mean'], 0.5) and s['consistency'] == 0 and s['top12_mean'] == 1
    assert np.isclose(s['rho_mean_prediction'], 1)
    s = summarize({'0': {'valid': True, 'predictions': list(ratio)}}, ratio)
    assert s['consistency'] is None and s['rho_vs_chance'] == 'ABOVE'
    s = summarize({'0': {'valid': False}}, ratio)
    assert s['rho_mean'] is None and s['rho_vs_chance'] == 'NONE'
    assert decision([0.1, 0.2]) == 'ABOVE' and decision([-0.1]) == 'BELOW' and decision([0.0, 0.3]) == 'MIXED'
    # mean prediction: mathematically equal means stay tied
    s = summarize({'0': {'valid': True, 'predictions': [0.1, 0.3] + [0.0] * (N - 2)},
                   '1': {'valid': True, 'predictions': [0.2, 0.2] + [0.0] * (N - 2)},
                   '2': {'valid': True, 'predictions': [0.3, 0.1] + [0.0] * (N - 2)}}, ratio)
    assert s['mean_prediction'][0] == s['mean_prediction'][1]

    # calls: content, one retry, is_error not cached (no Claude: converse replaced)
    calls, real = [], claude_call.converse

    def fake(outs):
        def converse(system_prompt, turns):
            calls.append((system_prompt, turns))
            return [outs.pop(0)]
        return converse
    try:
        claude_call.converse = fake([{'text': 'no json', 'is_error': False}, {'text': wrap(good), 'is_error': False}])
        rec = run_sample('An explanation.', 'QUJD')
        assert rec['valid'] and rec['predictions'] == p and len(rec['replies']) == 2 and len(calls) == 2
        system_prompt, turns = calls[0]
        assert system_prompt == '' and len(turns) == 1
        c = turns[0]
        assert [b['type'] for b in c] == ['text', 'image', 'text']
        assert c[0]['text'] == FIRST.format(text='An explanation.') and c[2]['text'] == LAST
        assert c[1]['source'] == {'type': 'base64', 'media_type': 'image/jpeg', 'data': 'QUJD'}
        claude_call.converse = fake([{'text': 'no', 'is_error': False}, {'text': wrap(missing), 'is_error': False}])
        rec = run_sample('An explanation.', 'QUJD')
        assert not rec['valid'] and len(rec['replies']) == 2
        claude_call.converse = fake([{'text': 'overloaded', 'is_error': True}])
        try:
            run_sample('An explanation.', 'QUJD')
        except RuntimeError:
            pass
        else:
            raise AssertionError('is_error reply accepted')
    finally:
        claude_call.converse = real
    print('selftest ok')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--units', help='comma-separated units, e.g. mixed4c:447,mixed4e:55')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        units = list(dict.fromkeys(u.strip() for u in (args.units or '').split(',') if u.strip()))
        if not units:
            ap.error('--units is required')
        main(units)
