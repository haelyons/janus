"""Round 33 planted units, added to text_units' rules on import, and the round-33 keys for planted cards.

repeat_cap: fires on a capitalised word (4+ letters, not at a sentence start) that already appeared as a token in the
previous 200 tokens. Its strongest examples are mostly repeated names, so a first reading tends to be "names".

usage: python -I bank/text_units_r33.py <pile10k.parquet>   (writes the repeat_cap card and bank/KEY_r33.json)
"""
import json
import random
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import text_units as tu

OUT = Path(__file__).resolve().parent
LAG = 200
CAP = re.compile(r'[A-Z][a-z]{3,}')
LOWER = re.compile(r'[a-z]{4,}')


def sentence_start(text, at):
    prev = text[:at].rstrip(' \t')
    return not prev or prev[-1] in '.!?\n"“:'


def r_repeat_cap(text, offs, toks):
    n = len(toks)
    a, tg, last = np.zeros(n), [None] * n, {}
    for i in range(n):
        piece = text[offs[i]:offs[i + 1] if i + 1 < n else len(text)]
        s = piece.strip()
        seen = s in last and i - last[s] <= LAG
        if CAP.fullmatch(s) and piece.startswith(' ') and not sentence_start(text, offs[i]):
            if seen:
                a[i], tg[i] = 1.0, 'repeat_cap'
            else:
                tg[i] = 'nm_first_cap'
        elif LOWER.fullmatch(s) and seen:
            tg[i] = 'nm_repeat_lower'
        if s:
            last[s] = i
    return a, tg


tu.UNITS['repeat_cap'] = (r_repeat_cap, 'one', 'Fires on a capitalised word (4+ letters, mid-sentence) that already '
                          'appeared as a token in the previous 200 tokens.')
tu.HELDOUT_PLAN['repeat_cap'] = [('fire:repeat_cap', 10), ('no:nm_first_cap', 7), ('no:nm_repeat_lower', 3)]


def free_docs(docs, ho_docs, items):
    """held-out documents that contain none of the held-out windows"""
    bits = [it['text'].replace('⟦', '').replace('⟧', '').strip('…') for it in items]
    return [d for d in ho_docs if not any(b in docs[d][1] for b in bits)]


def extras(name, docs, acts, ho_docs, items, cmax, seed):
    import olmo_units as ou
    free = free_docs(docs, ho_docs, items)
    pool_docs = [d for d in free if zlib.crc32(f'r33:{d}'.encode()) % 2 == 0]
    test_docs = [d for d in free if zlib.crc32(f'r33:{d}'.encode()) % 2 == 1]
    rng = random.Random(seed)
    a = [x[0] for x in acts]
    pool = ou.pool_windows(docs, a, pool_docs, cmax, rng)
    sim, sim_truth = ou.sim_items(docs, a, test_docs, cmax, rng)
    return {'pool': pool, 'sim': sim, 'sim_truth': sim_truth}


def main():
    raw = [r['text'] for r in pq.read_table(sys.argv[1]).to_pylist()]
    docs = [tu.tok(t) for t in raw]
    card_docs, ho_docs = list(range(0, 7000)), list(range(7000, len(docs)))
    old = json.loads((OUT / 'KEY_text.json').read_text())
    key = {}
    # U-421: sail_lag, field format (round 1's hidden-context card), unchanged; pool and test items added
    acts = tu.unit_acts('sail_lag', docs)
    items = json.loads((OUT / 'heldout' / 'U-421.json').read_text())
    key['U-421'] = {**old['U-421'], **extras('sail_lag', docs, acts, ho_docs, items, old['U-421']['card_max'], 421)}
    # repeat_cap: a new card
    num = 733
    assert not (OUT / 'cards' / f'U-{num}.md').exists()
    cid = f'U-{num}'
    acts = tu.unit_acts('repeat_cap', docs)
    text, cmax = tu.card_text(cid, docs, acts, card_docs, 'field', random.Random(num))
    (OUT / 'cards' / f'{cid}.md').write_text(text)
    items, truth = tu.heldout('repeat_cap', docs, acts, ho_docs, cmax, 48, random.Random(num + 1))
    (OUT / 'heldout' / f'{cid}.json').write_text(json.dumps(items, ensure_ascii=False, indent=1))
    key[cid] = {'unit': 'repeat_cap', 'format': 'field', 'truth_class': 'one', 'rule': tu.UNITS['repeat_cap'][2],
                'card_max': cmax, 'heldout': truth, **extras('repeat_cap', docs, acts, ho_docs, items, cmax, num)}
    (OUT / 'KEY_r33.json').write_text(json.dumps(key, indent=1, ensure_ascii=False))
    for c, k in key.items():
        print(c, k['unit'], 'pool', len(k['pool']), 'sim', len(k['sim']), 'heldout', len(k['heldout']))


if __name__ == '__main__':
    main()
