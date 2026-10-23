"""Round 30 unit selection by a fixed, seeded rule, from authors/round30_candidates.json (tools/r30_refs.py) and the
blind mapping authors/round30_mapping.reply.json (labels -> question relations and keywords). Written before any
round-30 data exists; no activation or answer is read here.

30A (referenced units):
- zoomin_4a: all four (mixed4a 234, 308, 325, 222).
- nick_inside: 4 drawn from unexamined car-circuit units whose label maps to at least one question.
- nick_outside: 3 drawn from unexamined car-circuit units whose label maps to none.
- equiv_variants: 1 drawn from mixed4b 411, 391, 434.  equiv_human_animal: 1 drawn from mixed4d 429, 427.
- anchors: mixed4b:431 (oriented, right) and mixed4b:409 (not oriented), round 29's units, rerun.
30B (random): 2 channels drawn from each band (mixed4a-4c, mixed4d-4e, mixed5a-5b), uniformly over the band's
channels, excluding examined units and 30A's.
Draws: random.Random(seed + k).sample(sorted pool, n), k the stratum's position in DRAWS; seed from r30_refs.

usage: python -I tools/r30_select.py
"""
import json
import random
from pathlib import Path

J = Path(__file__).resolve().parent.parent
CANDS = J / 'authors/round30_candidates.json'
REPLY = J / 'authors/round30_mapping.reply.json'
KEY = J / 'authors/round30_mapping_key.json'
OUT = J / 'authors/round30_selection.json'
NCH = {'mixed4a': 508, 'mixed4b': 512, 'mixed4c': 512, 'mixed4d': 528, 'mixed4e': 832, 'mixed5a': 832,
       'mixed5b': 1024}
ANCHORS = {'mixed4b:431': 'right', 'mixed4b:409': None}
DRAWS = [('nick_inside', 4), ('nick_outside', 3), ('equiv_variants', 1), ('equiv_human_animal', 1)]
BANDS = [('mixed4a', 'mixed4b', 'mixed4c'), ('mixed4d', 'mixed4e'), ('mixed5a', 'mixed5b')]
PER_BAND = 2


def check_reply(reply, key, questions):
    """every label answered; relations name a question 1-8 and its answers exactly"""
    assert set(reply) == set(key), sorted(set(key) ^ set(reply))
    for k, v in reply.items():
        assert isinstance(v.get('keywords'), list) and v['keywords'], k
        for r in v['relations']:
            ans = questions[r['question'] - 1]['answers']
            assert r['more'] and r['less'] and set(r['more']) <= set(ans) and set(r['less']) <= set(ans), (k, r)
            assert not set(r['more']) & set(r['less']) and isinstance(r['either'], bool), (k, r)


def main():
    cd = json.loads(CANDS.read_text())
    seed, cands, seen = cd['seed'], cd['candidates'], set(cd['examined'])
    key = json.loads(KEY.read_text())
    reply = json.loads(REPLY.read_text())
    questions = json.loads((J / 'authors/standing.reply.json').read_text())['questions']
    check_reply(reply, key, questions)
    by_text = {t: reply[k] for k, t in key.items()}
    refs = {}
    for c in cands:
        if c['examined']:
            continue
        m = by_text[c['label']]
        refs[c['unit']] = {**c, 'relations': m['relations'], 'keywords': [w.lower() for w in m['keywords']]}

    pools = {'nick_inside': [u for u, r in refs.items() if r['source'] == 'nick_car_circuit' and r['relations']],
             'nick_outside': [u for u, r in refs.items() if r['source'] == 'nick_car_circuit' and not r['relations']],
             'equiv_variants': [u for u, r in refs.items() if r['source'] == 'equiv_variants'],
             'equiv_human_animal': [u for u, r in refs.items() if r['source'] == 'equiv_human_animal']}
    strata = {'zoomin_4a': sorted(u for u, r in refs.items() if r['source'] == 'zoomin_4a')}
    for k, (name, n) in enumerate(DRAWS):
        strata[name] = sorted(random.Random(seed + k).sample(sorted(pools[name]), n))
    strata['anchors'] = list(ANCHORS)
    a_units = [u for s in strata.values() for u in s]
    assert len(a_units) == len(set(a_units)), a_units

    taken = seen | set(a_units)
    b_units = []
    for k, band in enumerate(BANDS):
        pool = sorted(f'{L}:{c}' for L in band for c in range(NCH[L]) if f'{L}:{c}' not in taken)
        b_units += sorted(random.Random(seed + 100 + k).sample(pool, PER_BAND))
    strata['random'] = b_units

    sel = {'seed': seed, 'pool_sizes': {k: len(v) for k, v in pools.items()}, 'strata': strata,
           'references': {u: refs[u] for u in a_units if u in refs},
           'anchors': {u: {'side': s} for u, s in ANCHORS.items()},
           'random_also_referenced': {u: refs[u] for u in b_units if u in refs}}
    OUT.write_text(json.dumps(sel, indent=1) + '\n')
    (J / 'authors/round30a_units.json').write_text(json.dumps(a_units) + '\n')
    (J / 'authors/round30b_units.json').write_text(json.dumps(b_units) + '\n')
    print('pool sizes', sel['pool_sizes'])
    for s, us in strata.items():
        for u in us:
            r = refs.get(u)
            print(f'{s:18s} {u:13s} ' + (f"{r['label'][:70]} | {r['relations']} | {r['keywords']}" if r else ''))


if __name__ == '__main__':
    main()
