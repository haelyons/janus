"""Round 36 analysis: growth metrics for the drawn units over the training sequence (PREDICTIONS.md, round 36).

Reads runs/r36/out/<KEY>.npz (tools/r36_record.py) in runs/r36/checkpoints.json order. The training sequence S is the
stage1, stage2 and main checkpoints in that order; f = main; SFT, DPO and Instruct are kept apart.
Definitions are the sealed ones: d_t = per-document max (pole +) or minus the per-document min (pole -); T_t = top 16
documents (ties to the lower document index); J = overlap / 16; settling s, core t0 and C, retention r and null r0
(next drawn unit of the same kind in the same layer, cyclic), width w, order o (needs reps).

usage: python -I tools/r36_analyze.py triples --model 7B          writes runs/r36/triples_7B.npy for r36_record reps
       python -I tools/r36_analyze.py metrics --model 7B [--reps runs/r36/reps_7B.npz]
                                                                    writes runs/r36/metrics_7B.json, prints the checks
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

J_ = Path(__file__).resolve().parent.parent
R = J_ / 'runs/r36'
BASE = {'7B': 'allenai/OLMo-2-1124-7B', '1B': 'allenai/OLMo-2-0425-1B'}
TOP, TOP_E = 16, 64


def sequence(model):
    seq = json.loads((R / 'checkpoints.json').read_text())[BASE[model]]
    S = [c for c in seq if c['phase'] in ('stage1', 'stage2', 'main')]
    post = {c['phase']: c for c in seq if c['phase'] in ('SFT', 'DPO', 'Instruct')}
    return S, post


def load(model, pole):
    """names, S, post, d[t] arrays [D, U] (S then post order), argmax positions likewise"""
    S, post = sequence(model)
    d, pos = [], []
    for c in S + list(post.values()):
        z = np.load(R / 'out' / f'{c["key"]}.npz')
        if pole == '+':
            d.append(z['u_max']), pos.append(z['u_argmax'])
        else:
            d.append(-z['u_min']), pos.append(z['u_argmin'])
        names = [str(x) for x in z['units']]
    return names, S, post, np.stack(d), np.stack(pos)


def topk(v, k):
    """the k highest entries, ties to the lower index"""
    return np.argsort(-v, kind='stable')[:k]


def jac(a, b):
    return len(set(a.tolist()) & set(b.tolist())) / TOP


def pct(v, docs):
    """median over docs of the percentile of v[doc] among all documents (half credit for ties)"""
    r = (rankdata(v, method='average') - 0.5) / len(v)
    return float(np.median(r[docs]))


def width(v):
    m = v.max()
    return int(np.sum(v >= 0.5 * m)) if m > 0 else None


def partner(names):
    """for each unit, the next drawn unit of the same kind in the same layer (cyclic)"""
    groups = {}
    for i, n in enumerate(names):
        l, kind = re.match(r'L(\d+):([nd])', n).groups()
        groups.setdefault((l, kind), []).append(i)
    out = {}
    for g in groups.values():
        for j, i in enumerate(g):
            out[i] = g[(j + 1) % len(g)]
    return out


def unit_metrics(d, nS):
    """per unit (column): J to f, settling index, core index and documents, widths. d [nT, D, U]"""
    f = nS - 1
    out = []
    for u in range(d.shape[2]):
        T = [topk(d[t, :, u], TOP) for t in range(d.shape[0])]
        Jf = [jac(T[t], T[f]) for t in range(nS)]
        s = next(i for i in range(nS) if all(x >= 0.5 for x in Jf[i:]))
        t0 = next((i for i in range(f) if jac(T[i], T[i + 1]) >= 0.5), None)
        out.append({'T': T, 'Jf': Jf, 's': s, 't0': t0, 'w': [width(d[t, :, u]) for t in range(nS)]})
    return out


def entry(tops, x):
    """earliest index in S from which document x stays in the unit's top 64 (tops: its top-64 sets over S)"""
    inside = [x in t for t in tops]
    return next(i for i in range(len(tops)) if all(inside[i:]))


def cmd_triples(model):
    names, S, post, d, pos = load(model, '+')
    nS, f = len(S), len(S) - 1
    M = unit_metrics(d, nS)
    trip = set()
    for u, m in enumerate(M):
        if m['t0'] is None:
            continue
        l = int(re.match(r'L(\d+)', names[u]).group(1))
        for x in m['T'][m['t0']]:
            trip.add((l, int(x), int(pos[m['t0'], x, u])))
        for x in m['T'][f]:
            trip.add((l, int(x), int(pos[f, x, u])))
    t = np.array(sorted(trip), np.int64)
    np.save(R / f'triples_{model}.npy', t)
    print('triples', len(t))


def order(d, pos, names, M, reps, nS):
    """o(u) per unit (None when undefined)"""
    f = nS - 1
    tri = np.load(R / f'triples_{reps[1]}.npy')
    Z = np.load(reps[0])
    idx = {tuple(map(int, r)): i for i, r in enumerate(tri)}
    mean, std = Z['mean'], np.maximum(Z['std'], 1e-6)

    def z(l, x, t, u):
        v = Z['reps'][idx[(l, int(x), int(pos[t, x, u]))]].astype(np.float64)
        return (v - mean[l]) / std[l]

    out = []
    for u, m in enumerate(M):
        if m['t0'] is None:
            out.append(None)
            continue
        l = int(re.match(r'L(\d+)', names[u]).group(1))
        C = m['T'][m['t0']]
        c = np.mean([z(l, x, m['t0'], u) for x in C], 0)
        xs = [x for x in m['T'][f] if x not in set(C.tolist())]
        tops = [set(topk(d[t, :, u], TOP_E).tolist()) for t in range(nS)]
        e = [entry(tops, x) for x in xs]
        dl = []
        for x in xs:
            v = z(l, x, f, u)
            dl.append(1 - float(v @ c / (np.linalg.norm(v) * np.linalg.norm(c))))
        out.append(float(spearmanr(e, dl)[0]) if len(xs) >= 5 and len(set(e)) > 1 else None)
    return out


def med(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return (float(np.median(v)) if v else None), len(v)


def cmd_metrics(model, reps):
    S, post = sequence(model)
    nS, f = len(S), len(S) - 1
    step = {c['step']: i for i, c in enumerate(S) if c['phase'] == 'stage1'}
    res = {'S': [c['revision'] for c in S], 'poles': {}}
    for pole in ('+', '-'):
        names, S, post, d, pos = load(model, pole)
        M = unit_metrics(d, nS)
        P = partner(names)
        kinds = {'neurons': [i for i, n in enumerate(names) if ':n' in n],
                 'directions': [i for i, n in enumerate(names) if ':d' in n]}
        o = order(d, pos, names, M, (reps, model), nS) if reps and pole == '+' else [None] * len(names)
        per = []
        for u, m in enumerate(M):
            v = P[u]
            r = pct(d[f, :, u], m['T'][m['t0']]) if m['t0'] is not None else None
            r0 = pct(d[f, :, u], M[v]['T'][M[v]['t0']]) if M[v]['t0'] is not None else None
            narrow = (m['w'][f] < m['w'][m['t0']]) if m['t0'] is not None and None not in (
                m['w'][f], m['w'][m['t0']]) else None
            row = {'unit': names[u], 'Jf': m['Jf'], 's': m['s'], 't0': m['t0'], 'r': r, 'r0': r0,
                   'w_t0': m['w'][m['t0']] if m['t0'] is not None else None, 'w_f': m['w'][f], 'narrow': narrow,
                   'o': o[u], 'rho_f': [float(spearmanr(d[t, :, u], d[f, :, u])[0]) for t in range(nS)]}
            for k, c in post.items():
                t = nS + list(post).index(k)
                row[f'rho_{k}'] = float(spearmanr(d[t, :, u], d[f, :, u])[0])
                row[f'J_{k}'] = jac(topk(d[t, :, u], TOP), m['T'][f])
            if 928646 in step and 600000 in step:
                row['J_last_stage1'] = jac(m['T'][step[600000]], m['T'][step[928646]])
                row['J_stage2'] = jac(m['T'][step[928646]], m['T'][f])
            per.append(row)
        summ = {}
        for kind, us in kinds.items():
            rows = [per[u] for u in us]
            s = {'n': len(rows),
                 'J_f_median': [med([r['Jf'][t] for r in rows])[0] for t in range(nS)],
                 'rho_f_median': [med([r['rho_f'][t] for r in rows])[0] for t in range(nS)],
                 's_median': med([r['s'] for r in rows]), 'cores': sum(r['t0'] is not None for r in rows),
                 'r_median': med([r['r'] for r in rows]), 'r0_median': med([r['r0'] for r in rows]),
                 'narrow_share': float(np.mean([r['narrow'] for r in rows if r['narrow'] is not None])),
                 'o_median': med([r['o'] for r in rows])}
            for k in post:
                s[f'rho_{k}_median'] = med([r[f'rho_{k}'] for r in rows])
                s[f'J_{k}_median'] = med([r[f'J_{k}'] for r in rows])
            if 'J_stage2' in rows[0]:
                s['J_last_stage1_median'] = med([r['J_last_stage1'] for r in rows])
                s['J_stage2_median'] = med([r['J_stage2'] for r in rows])
            layer = lambda r: int(re.match(r'L(\d+)', r['unit']).group(1))
            nl = max(layer(r) for r in rows) + 1
            s['s_median_by_layer'] = [med([r['s'] for r in rows if layer(r) == l])[0] for l in range(nl)]
            s['s_median_first_quarter'] = med([r['s'] for r in rows if layer(r) < nl // 4])
            s['s_median_last_quarter'] = med([r['s'] for r in rows if layer(r) >= nl - nl // 4])
            summ[kind] = s
        res['poles'][pole] = {'summary': summ, 'units': per}
    (R / f'metrics_{model}.json').write_text(json.dumps(res))
    check(model, res, step)


def check(model, res, step):
    S = res['S']
    n = res['poles']['+']['summary']['neurons']
    dr = res['poles']['+']['summary']['directions']
    at = lambda st: n['J_f_median'][step[st]]
    out = []
    if model == '7B':
        out.append(('P1', at(5000) is not None and at(5000) < 0.25, f'median J(5000, f) {at(5000)}'))
        out.append(('P2', n['s_median'][0] >= step[99000], f"median s index {n['s_median']} vs step 99000 index "
                    f"{step[99000]} ({S[int(np.floor(n['s_median'][0]))]})"))
        out.append(('P3', n['r_median'][0] >= 0.95 and n['r_median'][0] >= n['r0_median'][0] + 0.20,
                    f"r {n['r_median']} r0 {n['r0_median']} cores {n['cores']}/{n['n']}"))
        out.append(('P4', n['narrow_share'] > 0.5, f"narrow share {n['narrow_share']}"))
        out.append(('P5', n['o_median'][0] is not None and n['o_median'][0] <= 0.10, f"o {n['o_median']}"))
        out.append(('P6', n['r_median'][0] >= dr['r_median'][0] + 0.05,
                    f"neurons r {n['r_median']} directions r {dr['r_median']}"))
        out.append(('P7', n['rho_Instruct_median'][0] >= 0.90 and n['J_Instruct_median'][0] >= 0.5,
                    f"rho {n['rho_Instruct_median']} J {n['J_Instruct_median']}"))
        out.append(('P8', n['J_stage2_median'][0] <= n['J_last_stage1_median'][0],
                    f"J(928646, main) {n['J_stage2_median']} J(600000, 928646) {n['J_last_stage1_median']}"))
        out.append(('P9', n['s_median_first_quarter'][0] < n['s_median_last_quarter'][0],
                    f"layers 0-7 {n['s_median_first_quarter']} 24-31 {n['s_median_last_quarter']}"))
    else:
        out.append(('P10 P1', at(10000) < 0.25, f'median J(10000, f) {at(10000)}'))
        out.append(('P10 P3', n['r_median'][0] >= 0.95 and n['r_median'][0] >= n['r0_median'][0] + 0.20,
                    f"r {n['r_median']} r0 {n['r0_median']}"))
        out.append(('P10 P4', n['narrow_share'] > 0.5, f"narrow share {n['narrow_share']}"))
        out.append(('P10 P5', n['o_median'][0] is not None and n['o_median'][0] <= 0.10, f"o {n['o_median']}"))
        out.append(('P10 step0', at(0) <= 1 / 16, f'median J(0, f) {at(0)}'))
    for k, ok, note in out:
        print(k, 'HELD' if ok else 'FAILED', '|', note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('triples', 'metrics'))
    ap.add_argument('--model', choices=('7B', '1B'), required=True)
    ap.add_argument('--reps')
    a = ap.parse_args()
    cmd_triples(a.model) if a.cmd == 'triples' else cmd_metrics(a.model, a.reps)


if __name__ == '__main__':
    main()
