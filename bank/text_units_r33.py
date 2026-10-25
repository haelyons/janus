"""Round 33 planted units, added to text_units' rules on import, and the round-33 keys for planted cards.

repeat_cap: fires on a capitalised word (4+ letters, not at a sentence start) that already appeared as a token in the
previous 200 tokens. Its strongest examples are mostly repeated names, so a first reading tends to be "names".
num_plural (full round): fires on a lowercase word ending in s right after a number ("3 dogs"); its strongest
examples are mostly durations ("10 years"), so a first reading tends to be "time units".
paren_year (full round): fires on a year inside parentheses; its strongest examples are mostly citations
("Smith et al. (2009)"), so a first reading tends to be "citation years".

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


NUM = re.compile(r'\d[\d,]*')
PLURAL = re.compile(r'[a-z]{3,}s')
SINGULAR = re.compile(r'[a-z]{3,}')
YEAR = re.compile(r'(1[5-9]|20)\d\d')
NUMBER = re.compile(r'\d+')


def r_num_plural(text, offs, toks):
    n = len(toks)
    a, tg = np.zeros(n), [None] * n
    pieces = [text[offs[i]:offs[i + 1] if i + 1 < n else len(text)] for i in range(n)]
    for i in range(1, n):
        s, after_num = pieces[i].strip(), bool(NUM.fullmatch(pieces[i - 1].strip()))
        if PLURAL.fullmatch(s):
            a[i], tg[i] = (1.0, 'num_plural') if after_num else (0.0, 'nm_plural_no_num')
        elif after_num and SINGULAR.fullmatch(s):
            tg[i] = 'nm_num_singular'
    return a, tg


def in_parens(text, at):
    line = text[max(0, text.rfind('\n', 0, at) + 1):at]
    return line.rfind('(') > line.rfind(')')


def r_paren_year(text, offs, toks):
    n = len(toks)
    a, tg = np.zeros(n), [None] * n
    for i in range(n):
        s = text[offs[i]:offs[i + 1] if i + 1 < n else len(text)].strip()
        inside = in_parens(text, offs[i])
        if YEAR.fullmatch(s):
            a[i], tg[i] = (1.0, 'paren_year') if inside else (0.0, 'nm_year_outside')
        elif inside and NUMBER.fullmatch(s):
            tg[i] = 'nm_paren_number'
    return a, tg


tu.UNITS['num_plural'] = (r_num_plural, 'one', 'Fires on a lowercase word ending in s right after a number token.')
tu.HELDOUT_PLAN['num_plural'] = [('fire:num_plural', 10), ('no:nm_plural_no_num', 7), ('no:nm_num_singular', 3)]
tu.UNITS['paren_year'] = (r_paren_year, 'one', 'Fires on a year (1500-2099) inside an open parenthesis on its line.')
tu.HELDOUT_PLAN['paren_year'] = [('fire:paren_year', 10), ('no:nm_year_outside', 7), ('no:nm_paren_number', 3)]
tu.UNITS['repeat_cap'] = (r_repeat_cap, 'one', 'Fires on a capitalised word (4+ letters, mid-sentence) that already '
                          'appeared as a token in the previous 200 tokens.')
tu.HELDOUT_PLAN['repeat_cap'] = [('fire:repeat_cap', 10), ('no:nm_first_cap', 7), ('no:nm_repeat_lower', 3)]


def free_docs(docs, ho_docs, items):
    """held-out documents that contain none of the held-out windows"""
    bits = [it['text'].replace('⟦', '').replace('⟧', '').strip('…') for it in items]
    return [d for d in ho_docs if not any(b in docs[d][1] for b in bits)]


def extras(name, docs, acts, ho_docs, items, cmax, seed, sim_left=240):
    import olmo_units as ou
    free = free_docs(docs, ho_docs, items)
    pool_docs = [d for d in free if zlib.crc32(f'r33:{d}'.encode()) % 2 == 0]
    test_docs = [d for d in free if zlib.crc32(f'r33:{d}'.encode()) % 2 == 1]
    rng = random.Random(seed)
    a = [x[0] for x in acts]
    pool = ou.pool_windows(docs, a, pool_docs, cmax, rng)
    sim, sim_truth = ou.sim_items(docs, a, test_docs, cmax, rng, left=sim_left)
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
    # planted cards, built by text_units' rules; an existing card must come out unchanged
    for num, name in ((733, 'repeat_cap'), (518, 'num_plural'), (862, 'paren_year')):
        cid = f'U-{num}'
        acts = tu.unit_acts(name, docs)
        text, cmax = tu.card_text(cid, docs, acts, card_docs, 'field', random.Random(num))
        card = OUT / 'cards' / f'{cid}.md'
        assert not card.exists() or card.read_text() == text, cid
        card.write_text(text)
        items, truth = tu.heldout(name, docs, acts, ho_docs, cmax, 48, random.Random(num + 1))
        (OUT / 'heldout' / f'{cid}.json').write_text(json.dumps(items, ensure_ascii=False, indent=1))
        key[cid] = {'unit': name, 'format': 'field', 'truth_class': 'one', 'rule': tu.UNITS[name][2],
                    'card_max': cmax, 'heldout': truth, **extras(name, docs, acts, ho_docs, items, cmax, num)}
    (OUT / 'KEY_r33.json').write_text(json.dumps(key, indent=1, ensure_ascii=False))
    for c, k in key.items():
        print(c, k['unit'], 'pool', len(k['pool']), 'sim', len(k['sim']), 'heldout', len(k['heldout']))


if __name__ == '__main__':
    main()
