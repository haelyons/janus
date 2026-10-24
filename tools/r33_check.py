"""Round 33 pilot checks (PREDICTIONS.md, round 33). Written before any round-33 session exists.

  python -I tools/r33_check.py --cards U-421,U-733,O-...,O-... [--arms A0,F2_pool,P1_active,Y1_yoked] [--sample 1]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
from claude_call import parse_json  # noqa: E402


def keys():
    out = {}
    for f in ('KEY_r33.json', 'KEY_olmo.json'):
        p = J / 'bank' / f
        if p.exists():
            out.update(json.loads(p.read_text()))
    return out


def parsed(text, field):
    try:
        m = parse_json(text)
    except ValueError:
        return None
    return m if field in m else None


def score(rec, sim, k):
    replies = [r['text'] for r in rec['replies']]
    ans = next((m for m in (parsed(t, 'class') for t in reversed(replies[:-1])) if m), None)
    pred = parsed(replies[-1], 'predictions')
    got = {int(p['id']): bool(p['fires']) for p in (pred or {}).get('predictions', []) if 'id' in p and 'fires' in p}
    truth = k['heldout']
    acc = sum(got.get(t['id']) == t['fires'] for t in truth) / len(truth)
    by_tag = {}
    for t in truth:
        by_tag.setdefault(t['tag'], []).append(got.get(t['id']) == t['fires'])
    rho, n_sim = None, 0
    if sim:
        sp = parsed(sim['replies'][-1]['text'], 'predictions')
        pv = {int(p['id']): float(p['act']) for p in (sp or {}).get('predictions', []) if 'id' in p and 'act' in p}
        pairs = [(pv[t['id']], t['act']) for t in k['sim_truth'] if t['id'] in pv]
        n_sim = len(pairs)
        if n_sim >= 3 and len({p for p, _ in pairs}) > 1:
            rho = float(spearmanr([p for p, _ in pairs], [a for _, a in pairs])[0])
    return {'class': ans and ans.get('class'), 'explanation': ans and ans.get('explanation'), 'acc': acc,
            'by_tag': {t: round(float(np.mean(v)), 2) for t, v in by_tag.items()}, 'rho': rho, 'n_sim': n_sim}


def probes(rec):
    req = rec.get('requests') or []
    texts = [e for r in req for e in (r.get('show') or [])[:5] if isinstance(r, dict)]
    disc = [e for e in texts if isinstance(e, dict) and {'high', 'low'} <= set(map(str, e.get('predictions') or []))]
    return {'requests': len(req), 'texts': len(texts), 'discriminating': len(disc),
            'hypotheses': [r.get('hypotheses') for r in req if isinstance(r, dict)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cards', required=True)
    ap.add_argument('--arms', default='A0,F2_pool,P1_active,Y1_yoked')
    ap.add_argument('--sample', type=int, default=1)
    a = ap.parse_args()
    K, cards, arms = keys(), a.cards.split(','), a.arms.split(',')
    saved = sims_ok = 0
    res = {}
    for c in cards:
        for arm in arms:
            p = J / 'runs' / arm / c / f's{a.sample}.json'
            q = J / 'runs' / 'S0_sim' / arm / c / f's{a.sample}.json'
            if not p.exists():
                print(f'{c} {arm}: no session')
                continue
            saved += 1
            rec = json.loads(p.read_text())
            sim = json.loads(q.read_text()) if q.exists() else None
            s = score(rec, sim, K[c])
            sims_ok += s['n_sim'] >= 36
            if arm == 'P1_active':
                s['probes'] = probes(rec)
            res[(c, arm)] = s
            print(f'{c} {arm:10s} class {s["class"]}, held-out {s["acc"]:.2f} {json.dumps(s["by_tag"])}, '
                  f'sim rho {s["rho"] if s["rho"] is None else round(s["rho"], 2)} (n {s["n_sim"]})')
            print(f'    {s["explanation"]}')
            if 'probes' in s:
                print(f'    probes: {json.dumps(s["probes"])}')
    n = len(cards) * len(arms)
    print(f'Q1. sessions saved {saved} of {n}; simulator scores with 36+ of 40 predictions {sims_ok} of {n} ->',
          saved == n and sims_ok == n)
    pr = [res[(c, 'P1_active')]['probes'] for c in cards if (c, 'P1_active') in res]
    mean_req = float(np.mean([x['requests'] for x in pr])) if pr else None
    print(f'Q2. P1_active mean requests {mean_req} ->', mean_req is not None and mean_req >= 2)
    tot = sum(x['texts'] for x in pr)
    print(f'recorded: discriminating probe texts {sum(x["discriminating"] for x in pr)} of {tot}')
    for arm in arms:
        acc = [res[(c, arm)]['acc'] for c in cards if (c, arm) in res]
        rho = [res[(c, arm)]['rho'] for c in cards if (c, arm) in res and res[(c, arm)]['rho'] is not None]
        print(f'recorded: {arm:10s} mean held-out {np.mean(acc) if acc else None}, mean sim rho '
              f'{np.mean(rho) if rho else None}')


if __name__ == '__main__':
    main()
