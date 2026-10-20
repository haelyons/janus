"""Planted text units: exact rules over Pile-10k, shown to subjects as language-model neurons.

Each unit maps every GPT-2 token of a document to an activation by an exact rule, so the truth
about what the unit does is known. Writes subject-facing cards (bank/cards/), held-out items
(bank/heldout/) and the key (bank/KEY.json, orchestrator only).

usage: python -I bank/text_units.py <pile10k.parquet> [--selftest]
"""
import json
import zlib
import random
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken

ENC = tiktoken.get_encoding('gpt2')
MAXTOK = 2000
SCALE = 4.2  # raw activation units, so values look like a recorded neuron
OUT = Path(__file__).resolve().parent

# ---- lexicons -------------------------------------------------------------------------------
COFFEE = r'\b(coffee|espresso|cappuccino|latte|mocha|arabica|robusta|caffeine)\b'
CAFE = r'\b(caf[eé]s?|coffeehouse|coffee shop|Starbucks|barista)\b'
JAVA = r'\bJava\b'
CHEESE = r'\b(cheddar|brie|gouda|parmesan|mozzarella|feta|camembert|roquefort|gruy[eè]re|emmental|ricotta|mascarpone|halloumi|gorgonzola|provolone|manchego|pecorino|stilton|cheeses?)\b'
CPP = r'^#\s*(include|define)\b'
BIRDS = r'\b(sparrows?|finch(?:es)?|herons?|warblers?|pelicans?|pigeons?|falcons?|parrots?|penguins?|flamingos?|owls?|woodpeckers?|hummingbirds?|seagulls?|gulls?|storks?|kestrels?|vultures?|ducks?|geese|swans?)\b'
MONTH = r'\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)(?=\.?\s+\d{1,4}\b)'
SAIL = r'\b(sail\w*|yachts?|regattas?|masts?|hulls?|keels?|jib|starboard|helm|rudders?|anchor(?:ed|s|age)?|moor(?:ed|ing|ings)?|harbou?rs?|boats?|voyages?|ships?|vessels?|sailors?|decks?|mainsail|ashore|seafar\w*)\b'
LISTLINE = re.compile(r'^[ \t]*([-*•]|\d+[.)])[ \t]+\S[^\n]*$', re.M)

# near-miss lexicons for held-out items (never fire)
NM = {
    'drink': r'\b(tea|beer|wine|juice|milk|soda)\b',
    'lang': r'\b(Python|JavaScript|Ruby|Perl|Scala|Kotlin)\b',
    'food': r'\b(bread|butter|yogurt|pasta|rice|soup|tomato|chicken)\b',
    'ccode': r'\b(int|return|void|struct|printf|sizeof)\b',
    'animal': r'\b(dogs?|cats?|horses?|fish|cows?|rabbits?|sheep)\b',
    'monthfull': r'\b(January|February|March|April|June|July|August|September|October|November|December)(?=\s+\d{1,4}\b)',
}


def tok(doc):
    ids = ENC.encode_ordinary(doc)[:MAXTOK]
    for _ in range(4):  # the cut can split a multi-byte character; drop its tail
        try:
            text, offs = ENC.decode_with_offsets(ids)
            return ids, text, offs
        except UnicodeDecodeError:
            ids = ids[:-1]
    text = ENC.decode(ids)
    return tok(text[:-2])


def span_to_tok(offs, end):
    """index of the token containing character end-1 (the last char of a span)"""
    c = end - 1
    lo, hi = 0, len(offs) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if offs[mid] <= c:
            lo = mid
        else:
            hi = mid - 1
    return lo


def lex(text, offs, pat, flags=re.I):
    """every token covering a match (multi-token words fire on all their pieces)"""
    out = []
    for m in re.finditer(pat, text, flags):
        out += range(span_to_tok(offs, m.start() + 1), span_to_tok(offs, m.end()) + 1)
    return out


# ---- unit rules: return (acts, tags) per token; tags name the sub-rule that fired -------------
def r_dotnext(text, offs, toks):
    a, tg = np.zeros(len(toks)), [None] * len(toks)
    for i in range(len(toks) - 1):
        s = offs[i + 1]
        if text[s:s + 1] != '.':
            continue
        nxt = text[s + 1:s + 3]
        if nxt[:1] and (nxt[:1].isalnum() or nxt[:1] in '_/('):
            a[i], tg[i] = 1.0, 'dot_inword'           # www.x  3.14  obj.method
        elif len(nxt) == 2 and nxt[0] == ' ' and (nxt[1].islower() or nxt[1].isdigit() or nxt[1] == '('):
            a[i], tg[i] = 0.6, 'dot_abbrev'           # et al. (  Fig. 3  approx. ten
        elif not nxt or nxt[0] == '\n' or (nxt[0] == ' ' and nxt[1:2].isupper()):
            tg[i] = 'nm_sentence_end'
    return a, tg


def r_listend(text, offs, toks):
    a, tg = np.zeros(len(toks)), [None] * len(toks)
    list_lines = {m.end() for m in LISTLINE.finditer(text)}
    for m in LISTLINE.finditer(text):
        i = span_to_tok(offs, len(m.group(0).rstrip()) + m.start())
        bullet = m.group(1) in '-*•'
        a[i], tg[i] = (1.0, 'bullet') if bullet else (0.6, 'numbered')
        mid = span_to_tok(offs, m.start() + len(m.group(0)) // 2 + 1)
        if mid != i and tg[mid] is None:
            tg[mid] = 'nm_midline'
    for m in re.finditer(r'[^\n]{20,}$', text, re.M):
        if m.end() not in list_lines:
            i = span_to_tok(offs, len(m.group(0).rstrip()) + m.start())
            if tg[i] is None:
                tg[i] = 'nm_plain_line_end'
    return a, tg


def r_coffee(text, offs, toks):
    a, tg = np.zeros(len(toks)), [None] * len(toks)
    for i in lex(text, offs, CAFE):
        a[i], tg[i] = 0.7, 'cafe'
    for i in lex(text, offs, COFFEE):
        a[i], tg[i] = 1.0, 'coffee'
    for i in lex(text, offs, JAVA, 0):
        if tg[i] is None:
            a[i], tg[i] = 0.45, 'java'
    for k in ('drink', 'lang'):
        for i in lex(text, offs, NM[k], 0 if k == 'lang' else re.I):
            if tg[i] is None:
                tg[i] = f'nm_{k}'
    return a, tg


def r_cheese_cpp(text, offs, toks):
    a, tg = np.zeros(len(toks)), [None] * len(toks)
    for i in lex(text, offs, CHEESE):
        a[i], tg[i] = 1.0, 'cheese'
    for i in lex(text, offs, CPP, re.M):
        a[i], tg[i] = 0.65, 'cpp'
    for k in ('food', 'ccode'):
        for i in lex(text, offs, NM[k], re.I if k == 'food' else 0):
            if tg[i] is None:
                tg[i] = f'nm_{k}'
    return a, tg


def r_bird_month(text, offs, toks):
    a, tg = np.zeros(len(toks)), [None] * len(toks)
    for i in lex(text, offs, BIRDS):
        a[i], tg[i] = 1.0, 'bird'
    for i in lex(text, offs, MONTH, 0):
        a[i], tg[i] = 0.65, 'month'
    for k in ('animal', 'monthfull'):
        for i in lex(text, offs, NM[k], re.I if k == 'animal' else 0):
            if tg[i] is None:
                tg[i] = f'nm_{k}'
    return a, tg


def r_sail_lag(text, offs, toks):
    """punctuation, scaled by sailing terms 48-512 tokens back, only once none are in the last 48"""
    n = len(toks)
    a, tg = np.zeros(n), [None] * n
    hit = np.zeros(n)
    for i in lex(text, offs, SAIL):
        hit[i] = 1
    cs = np.concatenate([[0], np.cumsum(hit)])
    for i in range(n):
        piece = text[offs[i]:offs[i + 1] if i + 1 < n else len(text)].strip()
        if piece not in (',', '.', ';'):
            continue
        recent = cs[i + 1] - cs[max(0, i - 47)]
        far = cs[max(0, i - 47)] - cs[max(0, i - 511)]
        if recent == 0 and far >= 3:
            a[i], tg[i] = min(1.0, far / 12), 'lagged'
        elif recent == 0 and far == 0:
            tg[i] = 'nm_punct'
    return a, tg


UNITS = {
    'dotnext': (r_dotnext, 'one', 'Fires on the token right before a dot that does not end a sentence (www.x, 3.14, obj.method: 1.0; et al. (, Fig. 3: 0.6).'),
    'listend': (r_listend, 'one', 'Fires on the last token of a list-item line (bullets 1.0, numbered 0.6).'),
    'coffee': (r_coffee, 'one_plus_related', 'Core: coffee words (1.0). Offshoots: cafes/baristas (0.7); the word Java, mostly the programming language (0.45).'),
    'cheese_cpp': (r_cheese_cpp, 'unrelated', 'OR of two unrelated rules: cheese names (1.0); C preprocessor #include/#define (0.65).'),
    'bird_month': (r_bird_month, 'unrelated', 'OR of two unrelated rules: bird names (1.0); abbreviated month before a number (0.65).'),
    'sail_lag': (r_sail_lag, None, 'Fires on , . ; when 3+ sailing words appeared 48-512 tokens back and none in the last 48; scales with that count.'),
}


def h(*parts):
    return zlib.crc32(repr(parts).encode())


def noise(seed, n):
    return np.random.default_rng(seed).uniform(0.8, 1.0, n)


def unit_acts(name, docs):
    rule = UNITS[name][0]
    out = []
    for d, doc in enumerate(docs):
        toks, text, offs = doc
        a, tg = rule(text, offs, toks)
        a = a * noise(h(name, d), len(a))
        bg = np.random.default_rng(h(name, d, 'bg')).random(len(a)) < 0.003
        a = np.where((a == 0) & bg, np.random.default_rng(d).uniform(0, 0.06, len(a)), a) * SCALE
        out.append((a, tg))
    return out


def window(doc, i, left, right, marks=None, target=False):
    toks, text, offs = doc
    lo, hi = max(0, i - left), min(len(toks), i + right + 1)
    parts = []
    for j in range(lo, hi):
        piece = text[offs[j]:offs[j + 1] if j + 1 < len(toks) else len(text)]
        if target and j == i:
            parts.append(f'⟦{piece}⟧')
        elif marks is not None and marks[j] is not None:
            parts.append(f'«{piece}|{marks[j]}»')
        else:
            parts.append(piece)
    return ('…' if lo > 0 else '') + ''.join(parts) + ('…' if hi < len(toks) else '')


def pick_examples(acts, doc_ids, lo_frac, hi_frac, k, rng, mx):
    cands = []
    for d in doc_ids:
        a = acts[d][0]
        ok = np.where((a >= lo_frac * mx) & (a <= hi_frac * mx))[0]
        if len(ok):
            i = int(ok[np.argmax(a[ok])])
            cands.append((d, i, float(a[i])))
    rng.shuffle(cands)
    cands = cands[:k]
    return sorted(cands, key=lambda x: -x[2])


def card_text(cid, docs, acts, card_docs, fmt, rng):
    left = 48 if fmt == 'field' else 240
    mx = max(float(acts[d][0].max()) for d in card_docs)
    top = pick_examples(acts, card_docs, 0.75, 1.0, 12, rng, mx)
    mid = pick_examples(acts, card_docs, 0.3, 0.7, 8, rng, mx)
    cmax = max(x[2] for x in top)

    def marks(d):
        a = acts[d][0]
        if fmt == 'field':
            v = np.round(a / cmax * 10).astype(int)
            return [str(min(10, x)) if x >= 1 else None for x in v]
        return [f'{x:.2f}' if x >= 0.02 * cmax else None for x in a]

    scale_note = ('Activations are scaled 0-10 relative to the strongest example shown; only tokens at 1 or above are marked.'
                  if fmt == 'field' else
                  'Activations are raw recorded values; every token at 2% of the strongest shown or above is marked.')
    lines = [f'# Unit {cid}', '',
             'Recorded from one unit (a single neuron) in a language model, run over a broad sample of text '
             '(web pages, code, science abstracts, forums, law, Wikipedia).',
             f'Each example is a window of text around the token where the unit is most active in one document. '
             f'Marked tokens look like «token|activation». {scale_note}', '']
    if fmt == 'full':
        allv = np.concatenate([acts[d][0] for d in card_docs])
        lines += [f'Distribution over {len(allv):,} tokens: active (>5% of max) on {np.mean(allv > 0.05 * cmax) * 100:.2f}% of tokens; '
                  f'strongest value seen {cmax:.2f}. Windows show {left} tokens before and 15 after the peak token.', '']
    lines.append('## Strongest activations (one example per document)')
    for n, (d, i, v) in enumerate(top, 1):
        lines += ['', f'### Example {n}', window(docs[d], i, left, 15, marks(d))]
    lines += ['', '## Examples from the middle of the activation range (30-70% of the strongest)']
    if not mid:
        lines += ['', '(none found)']
    for n, (d, i, v) in enumerate(mid, 1):
        lines += ['', f'### Middle example {n}', window(docs[d], i, left, 15, marks(d))]
    return '\n'.join(lines) + '\n', cmax


HELDOUT_PLAN = {  # (tag or 'fire:<tag>' / 'no:<tag>', count)
    'dotnext': [('fire:dot_inword', 5), ('fire:dot_abbrev', 5), ('no:nm_sentence_end', 7), ('no:any', 3)],
    'listend': [('fire:bullet', 5), ('fire:numbered', 5), ('no:nm_plain_line_end', 5), ('no:nm_midline', 5)],
    'coffee': [('fire:coffee', 4), ('fire:cafe', 3), ('fire:java', 3), ('no:nm_drink', 4), ('no:nm_lang', 3), ('no:any', 3)],
    'cheese_cpp': [('fire:cheese', 5), ('fire:cpp', 5), ('no:nm_food', 5), ('no:nm_ccode', 5)],
    'bird_month': [('fire:bird', 5), ('fire:month', 5), ('no:nm_animal', 5), ('no:nm_monthfull', 5)],
    'sail_lag': [('fire:lagged', 10), ('no:nm_punct', 10)],
}


def heldout(name, docs, acts, ho_docs, cmax, left, rng):
    items = []
    for spec, k in HELDOUT_PLAN[name]:
        kind, tag = spec.split(':')
        pool = []
        for d in ho_docs:
            a, tg = acts[d]
            for i in range(len(a)):
                if kind == 'fire' and tg[i] == tag and a[i] >= 0.3 * cmax:
                    pool.append((d, i))
                elif kind == 'no' and a[i] <= 0.05 * cmax and (tg[i] == tag or (tag == 'any' and tg[i] is None and rng.random() < 0.01)):
                    pool.append((d, i))
        rng.shuffle(pool)
        seen, chosen = set(), []
        for d, i in pool:  # one item per document where possible
            if d not in seen:
                seen.add(d)
                chosen.append((d, i))
            if len(chosen) == k:
                break
        if len(chosen) < k:
            raise SystemExit(f'{name}: only {len(chosen)} held-out items for {spec}')
        items += [(d, i, kind == 'fire', tag) for d, i in chosen]
    rng.shuffle(items)
    return [{'id': n, 'text': window(docs[d], i, left, 15, target=True)} for n, (d, i, _, _) in enumerate(items, 1)], \
           [{'id': n, 'fires': f, 'tag': t, 'act': round(float(acts[d][0][i]), 3)} for n, (d, i, f, t) in enumerate(items, 1)]


def selftest():
    text = 'See www.example.com and Fig. 3 here. The end.\n- milk and eggs\n- bread\n'
    ids, text, offs = tok(text)
    a, tg = r_dotnext(text, offs, ids)
    fired = [text[offs[i]:offs[i + 1]] for i in range(len(ids) - 1) if a[i] > 0]
    assert fired == [' www', 'example', ' Fig'], fired
    a, tg = r_listend(text, offs, ids)
    fired = [text[offs[i]:offs[i + 1]] for i in range(len(ids) - 1) if a[i] > 0]
    assert fired == [' eggs', ' bread'], fired
    t2 = 'I like coffee at the café. Java is fine. Tea too.'
    ids, t2, offs = tok(t2)
    a, tg = r_coffee(t2, offs, ids)
    assert sorted(x for x in tg if x) == ['cafe', 'coffee', 'java', 'nm_drink'], tg
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    raw = [r['text'] for r in pq.read_table(sys.argv[1]).to_pylist()]
    docs = [tok(t) for t in raw]
    card_docs, ho_docs = list(range(0, 7000)), list(range(7000, len(docs)))
    rng = random.Random(20261020)
    ids = rng.sample(range(100, 999), 20)
    key = {}
    plan = [('dotnext', 'field'), ('listend', 'field'), ('coffee', 'field'), ('cheese_cpp', 'field'),
            ('bird_month', 'field'), ('sail_lag', 'field'), ('sail_lag', 'full')]
    (OUT / 'cards').mkdir(exist_ok=True)
    (OUT / 'heldout').mkdir(exist_ok=True)
    cache = {}
    for (name, fmt), num in zip(plan, ids):
        cid = f'U-{num}'
        acts = cache.setdefault(name, unit_acts(name, docs))
        text, cmax = card_text(cid, docs, acts, card_docs, fmt, random.Random(num))
        (OUT / 'cards' / f'{cid}.md').write_text(text)
        items, truth = heldout(name, docs, acts, ho_docs, cmax, 48 if fmt == 'field' else 240, random.Random(num + 1))
        (OUT / 'heldout' / f'{cid}.json').write_text(json.dumps(items, ensure_ascii=False, indent=1))
        tclass = UNITS[name][1] or ('cannot_tell' if fmt == 'field' else 'one')
        key[cid] = {'unit': name, 'format': fmt, 'truth_class': tclass, 'rule': UNITS[name][2], 'card_max': cmax, 'heldout': truth}
        print(cid, name, fmt, tclass, f'{len(text)} chars', flush=True)
    (OUT / 'KEY_text.json').write_text(json.dumps(key, indent=1))


if __name__ == '__main__':
    main()
