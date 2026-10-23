"""Round 31A checks (PREDICTIONS.md, round 31A). Written before any VLM answer, new photo or new table exists.

  python -I tools/r31_check.py validate --dir runs/r31/validate --models <name ...>          (model choice, V1-V3)
      expects <dir>/<name>_h0.json, <name>_h1.json, <name>_all.json (tools/r31_validate.py --out: Claude rounds 29,
      30B, 30A merged against the model, units of rounds 29, 30A, 30B, --half 0, 1, all), <dir>/bench_claude.json
      (Claude round 29 against round 30A, round-29 units, --half all) and runs/r31/answers/val_<name>.json
  python -I tools/r31_check.py tables --new runs/r31/tables                                   (P1-P6, step 2)
"""
import argparse
import json
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
OLD = {u: p for p in (J / 'runs/pairtable', J / 'runs/r30a/pairtable', J / 'runs/r30b/pairtable')
       for u in [d.name for d in p.glob('mixed*')]}
ORIENTED = {'mixed4b:431': 'right', 'mixed4b:443': 'right', 'mixed4b:429': 'left', 'mixed4b:384': 'left'}
UNORIENTED = ['mixed4b:409', 'mixed4b:418']
ZOOMIN = ['mixed4a:222', 'mixed4a:234', 'mixed4a:308', 'mixed4a:325']
CAR_SIDE = ['mixed5a:160', 'mixed5a:23', 'mixed5b:573', 'mixed5b:907']
RANDOM = json.loads((J / 'authors/round30b_units.json').read_text())
TOWARD = 'toward or away from the camera'


def up(c):
    return c['d'] is not None and c['d'] > 0 and c['pct'] is not None and c['pct'] >= 95


def corr(x, y):
    k = [i for i in range(len(x)) if x[i] is not None and y[i] is not None]
    if len(k) < 3:
        return None
    a, b = np.asarray([x[i] for i in k]), np.asarray([y[i] for i in k])
    return None if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def facing_sides(st):
    """the favoured sides of the held every-arm rows comparing Q1 left against right"""
    return sorted({p['first_answer'] for p in st['picks']['every'] if p['q'] == 0 and p['T']['outcome'] == 'held'
                   and {p['first_answer'], p['second_answer']} == {'left', 'right'}})


def held_rows(st):
    return [p for p in st['picks']['every'] if p['T']['outcome'] == 'held']


def tables(new):
    S = {d.name: json.loads((d / 'stats.json').read_text()) for d in sorted(Path(new).glob('mixed*'))}
    ev = [p for st in S.values() for p in st['picks']['every']]
    hi = [p for p in ev if p['D']['pct'] >= 95]
    p1 = sum(up(p['T']) for p in hi) / len(hi) if hi else None
    p1n = sum(up(p['T_null']) for p in hi) / len(hi) if hi else None
    print(f'P1. rows at 95+ on D: {len(hi)}; at 95+ on T {p1}; T_null {p1n} ->',
          p1 is not None and p1 >= 0.5 and p1n <= 0.10)
    r_all = corr([p['D']['d'] for p in ev], [p['T']['d'] for p in ev])
    q4 = [p for p in ev if p['q'] == 3]
    r_q4 = corr([p['D']['d'] for p in q4], [p['T']['d'] for p in q4])
    print(f'P2. split-half r of d: all rows ({len(ev)}) {r_all}; Q4 rows ({len(q4)}) {r_q4} ->',
          r_all is not None and r_all >= 0.70 and r_q4 is not None and r_q4 >= 0.30)
    sides = {u: facing_sides(S[u]) for u in list(ORIENTED) + UNORIENTED + ZOOMIN if u in S}
    p3 = all(sides.get(u) == [s] for u, s in ORIENTED.items()) and all(not sides.get(u) for u in UNORIENTED)
    print(f'P3. facing sides held: {json.dumps({u: sides.get(u) for u in list(ORIENTED) + UNORIENTED})} ->', p3)
    z = {u: sides.get(u) for u in ZOOMIN}
    one = all(v is not None and len(v) == 1 for v in z.values())
    p4 = one and z['mixed4a:234'] == z['mixed4a:308'] and z['mixed4a:325'] == z['mixed4a:222'] \
        and z['mixed4a:308'] != z['mixed4a:222']
    print(f'P4. Zoom In 4a facing sides held: {json.dumps(z)} ->', p4)
    side = {u: [p['row'] for p in held_rows(S[u]) if p['q'] == 0 and p['first_answer'] in ('left', 'right')
                and p['second_answer'] == TOWARD] for u in CAR_SIDE if u in S}
    print(f'P5. car side-view units with a held side-over-toward row: {json.dumps(side)} ->',
          sum(bool(v) for v in side.values()) <= 1)
    dist = {u: [p['row'] for p in held_rows(S[u]) if p['T']['distinctive']] for u in RANDOM if u in S}
    print(f'P6. random units with a held distinctive row: {json.dumps(dist)} ->',
          sum(bool(v) for v in dist.values()) >= 3)
    if 'mixed4b:12' in S:
        print('recorded: 4b:12 held position rows:', [(p['row'], p['first_answer'], p['second_answer'])
                                                      for p in held_rows(S['mixed4b:12']) if p['q'] == 3])
    if 'mixed4d:427' in S:
        print('recorded: 4d:427 facing rows:', [(p['first_answer'], p['second_answer'], round(p['D']['pct']),
                                                p['T']['pct'] and round(p['T']['pct']), p['T']['outcome'])
                                               for p in S['mixed4d:427']['picks']['every'] if p['q'] == 0])
    print('recorded: held every-arm rows per unit, old pool -> new pool, and photos (D + T):')
    for u, st in S.items():
        old = json.loads((OLD[u] / u / 'stats.json').read_text()) if u in OLD else None
        o = f'{len(held_rows(old))} ({len(old["D"]) + len(old["T"])})' if old else '-'
        print(f'  {u:13s} {o:>10s} -> {len(held_rows(st))} ({len(st["D"]) + len(st["T"])})')


def facing_row_pct(u, side, answers):
    """V3: the D-table percentile of the row "side over the other side" (Q1) for unit u, the table built as
    pairtable.run_unit builds its real table, with the given answers ({photo id (str): record}; a missing or invalid
    record a row of None); None when the table has no such row"""
    import sys
    sys.path.insert(0, str(J / 'tools'))
    import contrastloop
    import pairtable
    import proploop
    questions = pairtable.load_questions()
    names = json.loads(proploop.NAMES.read_text())
    with np.load(proploop.ACTS) as z:
        labels = np.asarray(z['classes'], dtype=np.int64)
    rec = proploop.parse_unit(u)
    V = proploop.layer_values([rec['layer']], None, np.arange(proploop.N_PHOTOS))[rec['layer']]
    _, _, D, _ = pairtable.unit_photos(rec, V, labels, names)
    peers = [p for p, _ in contrastloop.peers(V, labels, rec['chan'])]
    nq = len(questions)
    rows = []
    for i in D:
        r = answers.get(str(i))
        rows.append(r['answers'] if r and r.get('valid') else [None] * nq)
    D_ids = np.asarray(D, dtype=np.int64)
    table = pairtable.table_rows(proploop.centre(V[D_ids], labels[D_ids]), rows, questions, rec['chan'], peers)
    a = questions[0]['answers']
    other = 'left' if side == 'right' else 'right'
    got = [r['pct'] for r in table if r['q'] == 0 and a[r['first']] == side and a[r['second']] == other]
    return got[0] if got else None


def validate(d, models):
    d = Path(d)
    R = {m: {h: json.loads((d / f'{m}_{h}.json').read_text()) for h in ('h0', 'h1', 'all')} for m in models}
    bench = json.loads((d / 'bench_claude.json').read_text())
    mean0 = {m: float(np.mean([x['rate'] for x in R[m]['h0']['question_agreement']])) for m in models}
    sel = max(models, key=lambda m: (mean0[m], -models.index(m)))
    print('calibration half, mean agreement with Claude:', json.dumps({m: round(v, 4) for m, v in mean0.items()}),
          '-> selected', sel)
    b = [x['rate'] for x in bench['question_agreement']]
    c = [x['rate'] for x in R[sel]['h1']['question_agreement']]
    ok = [ci is not None and bi is not None and ci >= bi - 0.05 for ci, bi in zip(c, b)]
    v1 = sum(ok) >= 6 and ok[0] and ok[1]
    print('V1. confirmation half, per question (model, benchmark):',
          [(round(ci, 3), round(bi, 3)) for ci, bi in zip(c, b)], 'within 5 points:', ok, '->', v1)
    rd, rb = R[sel]['all']['summary']['r_d'], bench['summary']['r_d']
    v2 = rd is not None and rb is not None and rd >= 0.8 * rb
    print(f'V2. table r(d): model {rd}, Claude against Claude {rb}; bar {None if rb is None else 0.8 * rb} ->', v2)
    ans = json.loads((J / f'runs/r31/answers/val_{sel}.json').read_text())
    pcts = {u: facing_row_pct(u, s, ans) for u, s in ORIENTED.items()}
    v3 = sum(p is not None and p >= 95 for p in pcts.values()) >= 3
    print('V3. oriented units, preferred-side facing row percentile on D:', json.dumps(pcts), '->', v3)
    for m in models:
        print(f'recorded: {m}: h1 rates {[round(x["rate"], 3) for x in R[m]["h1"]["question_agreement"]]}; '
              f'r(d) all {R[m]["all"]["summary"]["r_d"]}; photos {R[m]["all"]["photos"]}')
    print('step 2:', 'run, with ' + sel if v1 and v2 and v3 else 'not run')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('tables')
    t.add_argument('--new', default=str(J / 'runs/r31/tables'))
    v = sub.add_parser('validate')
    v.add_argument('--dir', default=str(J / 'runs/r31/validate'))
    v.add_argument('--models', nargs='+', required=True)
    a = ap.parse_args()
    if a.cmd == 'tables':
        tables(a.new)
    else:
        validate(a.dir, a.models)


if __name__ == '__main__':
    main()
