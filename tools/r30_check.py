"""Round 30 prediction checks and failure attribution (PREDICTIONS.md, round 30), on runs/r30<stage>/ and
authors/round30_selection.json. Written before any round-30 result exists.

usage: python -I tools/r30_check.py --stage a|b
"""
import argparse
import csv
import json
import re
from pathlib import Path

J = Path(__file__).resolve().parent.parent
SEL = J / 'authors/round30_selection.json'
AGENT_REAL = ['real_s1', 'real_s2', 'real_s3']
AGENT_NULL = ['null_s1', 'null_s2', 'null_s3']
PICKING = AGENT_REAL + AGENT_NULL + ['rule_real', 'rule_null']
BARE = ['bare_s1', 'Q0_s1', 'F1_s1']
SENSITIVE, INSENSITIVE = 2.5, 10.0  # min mirror retention percentile over strata; calibrated on round 29's 6 units
SIDE = re.compile(r'\b(left|right)\b', re.I)
ZOOMIN_SAME = [('mixed4a:234', 'mixed4a:308'), ('mixed4a:325', 'mixed4a:222')]
ZOOMIN_DIFF = [('mixed4a:308', 'mixed4a:222')]


def ans(st, q, i):
    return st['questions'][q]['answers'][i].strip().lower()


def lr_side(st, r):
    """the side a facing (Q1 left vs right) or position (Q4 left third vs right third) row favours, else None"""
    a, b = ans(st, r['q'], r['first']), ans(st, r['q'], r['second'])
    if r['q'] == 0 and {a, b} == {'left', 'right'}:
        return a
    if r['q'] == 3 and {a, b} == {'left third', 'right third'}:
        return a.split()[0]
    return None


def held(p, key='T'):
    return p[key]['outcome'] == 'held'


def rel_match(st, r, rel, contrary=False):
    if r['q'] != rel['question'] - 1:
        return False
    a, b = ans(st, r['q'], r['first']), ans(st, r['q'], r['second'])
    more, less = {x.lower() for x in rel['more']}, {x.lower() for x in rel['less']}
    fwd, back = a in more and b in less, a in less and b in more
    if contrary:
        return back and not rel['either']
    return fwd or (back and rel['either'])


def mirror_min(unit, stage):
    layer = unit.split(':')[0]
    for f in (J / f'runs/r30{stage}/rangeprofile/{layer}.csv', J / f'runs/rangeprofile/{layer}.csv'):
        if f.exists():
            for row in csv.DictReader(f.open()):
                if row['unit'] == unit:
                    v = [float(row[f'mirror_ret_pct_s{s}']) for s in (1, 2, 3) if row.get(f'mirror_ret_pct_s{s}')
                         not in (None, '', 'nan')]
                    v = [x for x in v if x == x]
                    return min(v) if v else None
    return None


def mirror_class(m):
    if m is None:
        return 'no profile'
    return 'sensitive' if m <= SENSITIVE else 'insensitive' if m >= INSENSITIVE else 'between'


def keyword_hit(text, kws):
    t = (text or '').lower()
    return any(re.search(r'\b' + re.escape(k), t) for k in kws)


def rate(picks, key='T'):
    chk = [p for p in picks if p[key]['outcome'] != 'too few']
    return sum(held(p, key) for p in chk), len(chk)


def unit_report(u, st, bare, fin, ref, stage):
    """per-unit record and printed block"""
    picks = st['picks']
    every = picks['every']
    real_rows = st['tables']['real']
    rec = {'unit': u}
    lr_every = sorted({lr_side(st, p) for p in every if lr_side(st, p) and held(p)})
    lr_every_facing = sorted({lr_side(st, p) for p in every if lr_side(st, p) and p['q'] == 0 and held(p)})
    rec['lr_held_every'], rec['facing_side_every'] = lr_every, lr_every_facing
    rec['lr_held_picking'] = sorted({(k, lr_side(st, p)) for k in PICKING for p in picks.get(k, [])
                                     if lr_side(st, p) and held(p)})
    rec['mirror_min'] = mirror_min(u, stage)
    rec['mirror'] = mirror_class(rec['mirror_min'])
    ex = {k: (st['readings'].get(k) or {}).get('explanation') for k in AGENT_REAL}
    bx = {k: (bare.get(k) or {}).get('explanation') for k in BARE}
    rec['side_named_agent'] = sum(bool(e and SIDE.search(e)) for e in ex.values())
    rec['side_named_bare'] = sum(bool(e and SIDE.search(e)) for e in bx.values())
    ar = [p for k in AGENT_REAL for p in picks.get(k, [])]
    rec['agent_real_held'], rec['agent_real_n'] = rate(ar)
    rec['agent_real_distinctive'] = sum(p['T']['distinctive'] for p in ar)
    rec['agent_samples_with_held_distinctive'] = sum(any(p['T']['distinctive'] for p in picks.get(k, []))
                                                     for k in AGENT_REAL)
    rec['strict_distinctive_rows'] = sorted({p['row'] for p in ar if held(p) and p['T']['peer_pct']
                                             and max(p['T']['peer_pct']) < 95})
    if ref is not None:
        rels = ref['relations']
        rec['keywords_agent'] = sum(keyword_hit(e, ref['keywords']) for e in ex.values())
        rec['keywords_bare'] = sum(keyword_hit(e, ref['keywords']) for e in bx.values())
        if rels:
            m = lambda r, c=False: any(rel_match(st, r, x, c) for x in rels)  # noqa: E731
            ev_hit = [p['row'] for p in every if m(p) and held(p)]
            d_high = [r['row'] for r in real_rows if m(r) and r['pct'] >= 95]
            tab = [r['row'] for r in real_rows if m(r)]
            agent_hits = sum(any(m(p) and held(p) for p in picks.get(k, [])) for k in AGENT_REAL)
            rule_hit = any(m(p) and held(p) for p in picks['rule_real'])
            contra = [p['row'] for p in every if m(p, True) and held(p)]
            lr_rel = any(x['question'] in (1, 4) and ({a.lower() for a in x['more'] + x['less']} & {
                'left', 'right', 'left third', 'right third'}) for x in rels)
            if ev_hit:
                cause = 'success' if agent_hits >= 2 else 'interpretation: missed'
            elif d_high:
                cause = 'measurement: did not replicate on T'
            elif not tab:
                cause = 'measurement: too few photos for the reference answers'
            elif lr_rel and rec['mirror'] == 'sensitive':
                cause = 'measurement: low on both halves, intervention supports the reference'
            elif lr_rel and rec['mirror'] == 'insensitive':
                cause = 'reference not supported: low on both halves and by the intervention'
            else:
                cause = 'unresolved: low on both halves'
            if contra:
                cause += '; reference contradicted'
            rec.update(ref_rows_every_held=ev_hit, ref_rows_D95=d_high, ref_rows_table=tab, agent_hits=agent_hits,
                       rule_hit=rule_hit, contrary_held=contra, cause=cause)
        else:
            rec['cause'] = 'missing candidate (no question fits the reference)'
    if fin:
        add = []
        ev = {p['row']: p for p in every}
        for k, r in fin.items():
            if k == 'unit' or not r.get('valid'):
                continue
            for row in set(r['rows']) - set(r['picked']):
                add.append((k, row, ev[row]['T']['outcome'] if row in ev else 'no check'))
        rec['final_unchecked_additions'] = sorted(add)
    print(f'== {u} mirror {rec["mirror"]} ({rec["mirror_min"]}) peers '
          + ', '.join(f'{p["unit"]} ({p["corr"]:.2f})' for p in st['peers']))
    for k in AGENT_REAL + AGENT_NULL + ['rule_real']:
        rd = (st['readings'].get(k) or {}).get('explanation') or ''
        ps = [(p['row'], ans(st, p['q'], p['first'])[:18], ans(st, p['q'], p['second'])[:18], p['T']['outcome'],
               p['T']['distinctive']) for p in picks.get(k, [])]
        print(f'  {k}: {ps} | {rd[:140]}')
    for k in BARE:
        print(f'  {k}: {(bx[k] or "")[:160]}')
    print('  ' + json.dumps({k: v for k, v in rec.items() if k != 'unit'}))
    return rec


def answer_agreement(stage):
    old, new = J / 'runs/pairtable/answers.json', J / f'runs/r30{stage}/pairtable/answers.json'
    if not (old.exists() and new.exists()):
        return None
    rows = []
    for f in (old, new):
        d = {}
        for r in json.loads(f.read_text()).values():
            if not r['failed']:
                d.update({i: a for i, a in zip(r['photos'], r['answers'])})
        rows.append(d)
    common = sorted(set(rows[0]) & set(rows[1]))
    out = []
    for q in range(8):
        pairs = [(rows[0][i][q], rows[1][i][q]) for i in common if rows[0][i][q] is not None
                 and rows[1][i][q] is not None]
        out.append((sum(a == b for a, b in pairs), len(pairs)))
    return len(common), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['a', 'b'])
    a = ap.parse_args()
    sel = json.loads(SEL.read_text())
    strata = sel['strata']
    units = json.loads((J / f'authors/round30{a.stage}_units.json').read_text())
    refs = {**sel['references'], **sel.get('random_also_referenced', {})}
    R = J / f'runs/r30{a.stage}'
    recs, S = {}, {}
    for u in units:
        st = json.loads((R / 'pairtable' / u / 'stats.json').read_text())
        S[u] = st
        bp, fp = R / 'bare' / u / 'readings.json', R / 'finalread' / u / 'stats.json'
        bare = json.loads(bp.read_text()) if bp.exists() else {}
        fin = json.loads(fp.read_text()) if fp.exists() else None
        recs[u] = unit_report(u, st, bare, fin, refs.get(u), a.stage)
    stratum = {u: s for s, us in strata.items() for u in us}

    # calibration (item 4 on 30A, item 9 on 30B)
    ar = [p for u in units for k in AGENT_REAL for p in S[u]['picks'].get(k, [])]
    an = [p for u in units for k in AGENT_NULL for p in S[u]['picks'].get(k, [])]
    allp = ar + an + [p for u in units for k in ('rule_real', 'rule_null') for p in S[u]['picks'][k]]
    (hr, nr), (hn, nn), (hs, ns) = rate(ar), rate(an), rate(allp, 'T_null')
    fr, fn, fs = (h / n if n else float('nan') for h, n in ((hr, nr), (hn, nn), (hs, ns)))
    calib = fn <= 0.15 and fs <= 0.10 and fr >= 2 * fn
    print(f'calibration: agent real {hr}/{nr} ({fr:.2f}), agent null {hn}/{nn} ({fn:.2f}), shuffled T {hs}/{ns} '
          f'({fs:.2f}) -> {calib}')

    # mirror control (item 5), both stages, units with a profile
    ins = [u for u in units if recs[u]['mirror'] == 'insensitive']
    bad = [(u, recs[u]['lr_held_picking']) for u in ins if recs[u]['lr_held_picking']]
    sen = [u for u in units if recs[u]['mirror'] == 'sensitive']
    sen_hit = [u for u in sen if recs[u]['lr_held_every']]
    ok5 = not bad and (not sen or len(sen_hit) >= len(sen) / 2)
    print(f'5. mirror-insensitive units with a held left/right row in a picking arm: {bad} (of {ins}); '
          f'mirror-sensitive with a held left/right row in every: {sen_hit} of {sen} -> {ok5}')

    if a.stage == 'a':
        r431, r409 = recs['mixed4b:431'], recs['mixed4b:409']
        hits431 = sum(any(lr_side(S['mixed4b:431'], p) == 'right' and p['q'] == 0 and held(p) for p in S['mixed4b:431']['picks'].get(k, [])) for k in AGENT_REAL)
        ok1 = hits431 >= 2 and not r409['lr_held_picking']
        print(f'1. anchors: 431 right-facing row held in {hits431} of 3 agent samples; 409 held left/right rows in '
              f'picking arms {r409["lr_held_picking"]} -> {ok1}')
        side = {u: recs[u]['facing_side_every'] for u in strata['zoomin_4a']}
        one = {u: s[0] for u, s in side.items() if len(s) == 1}
        viol = [(x, y, 'same') for x, y in ZOOMIN_SAME if x in one and y in one and one[x] != one[y]]
        viol += [(x, y, 'differ') for x, y in ZOOMIN_DIFF if x in one and y in one and one[x] == one[y]]
        ok2 = len(one) >= 2 and not viol
        print(f'2. Zoom In 4a facing sides (held in every): {side}; violations {viol} -> {ok2}')
        ins_n = strata['nick_inside']
        good = [u for u in ins_n if recs[u].get('agent_hits', 0) >= 2]
        print(f'3. nick_inside with a held reference row in >= 2 of 3 agent samples: {good} of {ins_n} -> '
              f'{len(good) >= 2}')
        print(f'4. calibration -> {calib}')
        outs = strata['nick_outside'] + strata['equiv_variants']
        kb = [u for u in outs if recs[u].get('keywords_bare', 0) >= 1]
        ka = [u for u in outs if recs[u].get('keywords_agent', 0) >= 1]
        print(f'6. outside units, reference keyword in a bare/Q0/F1 reading: {kb}; in an agent explanation: {ka} of '
              f'{outs} -> {len(kb) >= 3 and len(ka) < len(kb)}')
        uj = [u for u in units if not recs[u]['lr_held_every'] and recs[u]['side_named_agent']]
        ujb = [u for u in units if not recs[u]['lr_held_every'] and recs[u]['side_named_bare']]
        print(f'7. units with no held left/right row whose agent explanations name a side: {uj} (bare: {ujb}) -> '
              f'{len(uj) >= 1}')
        inside = [u for u in units if stratum[u] in ('zoomin_4a', 'nick_inside', 'equiv_human_animal')
                  and refs.get(u, {}).get('relations')]
        causes = {u: recs[u]['cause'] for u in inside}
        miss = [u for u, c in causes.items() if c.startswith('interpretation')]
        print(f'8. inside causes: {json.dumps(causes, indent=1)}\n   interpretation failures {miss} -> {len(miss) <= 1}')
        print('   outside causes: ' + json.dumps({u: recs[u].get('cause') for u in outs}))
        ag = answer_agreement('a')
        if ag:
            print(f'answer agreement with round 29 on {ag[0]} common photos, per question: '
                  + ', '.join(f'Q{q + 1} {h}/{n}' for q, (h, n) in enumerate(ag[1])))
    else:
        got = [u for u in units if recs[u]['agent_samples_with_held_distinctive'] >= 2]
        print(f'9. calibration on random units -> {calib}')
        print(f'10. random units with a held, distinctive agent pick in >= 2 of 3 samples: {got} of {units} -> '
              f'{len(got) >= 3}')
    (R / 'r30_check.json').write_text(json.dumps(recs, indent=1) + '\n')


if __name__ == '__main__':
    main()
