"""Score saved runs against the key: class against truth, held-out accuracy by item type.

usage: python -I tools/score.py  (writes runs/scores.json and prints a compact table)
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claude_call import parse_json

J = Path(__file__).resolve().parent.parent
KEY = {**json.loads((J / 'bank/KEY_text.json').read_text())}
if (J / 'bank/KEY_img.json').exists():
    KEY.update(json.loads((J / 'bank/KEY_img.json').read_text()))
for v in ('v2', 'v3'):
    if (J / f'bank/KEY_img_{v}.json').exists():
        KEY.update(json.loads((J / f'bank/KEY_img_{v}.json').read_text()))
ARMS = ['A0_bare', 'C0_field', 'C1_super', 'C2_clean', 'C3_naive', 'C4_informed', 'P0_probe', 'N0_nouns']
OK = {'one': {'one', 'one_plus_related'}, 'one_plus_related': {'one', 'one_plus_related'},
      'unrelated': {'unrelated'}, 'cannot_tell': {'cannot_tell', 'one', 'one_plus_related'}}


def score_run(rec, key):
    cls = (rec.get('turn1') or {}).get('class')
    t = key['truth_class']
    preds = {p.get('id'): p.get('fires') for p in (rec.get('turn2') or {}).get('predictions', [])}
    by_tag = collections.defaultdict(list)
    for h in key['heldout']:
        by_tag[h['tag']].append(preds.get(h['id']) == h['fires'])
    allv = [v for vs in by_tag.values() for v in vs]
    return {'class': cls, 'class_ok': (cls in OK[t]) if t else None,
            'heldout_acc': sum(allv) / len(allv) if allv else None,
            'by_tag': {k: sum(v) / len(v) for k, v in by_tag.items()},
            'explanation': (rec.get('turn1') or {}).get('explanation')}


def main():
    rows = []
    for f in sorted(J.glob('runs/*/*/s*.json')):
        rec = json.loads(f.read_text())
        for n, r in enumerate(rec['replies'], 1):  # re-parse from the saved reply text
            try:
                rec[f'turn{n}'] = parse_json(r['text'])
            except ValueError:
                rec[f'turn{n}'] = None
        if 'requests' in rec:  # probe arm: the answer and the held-out predictions are the last two replies
            rec['turn1'], rec['turn2'] = [parse_json(r['text']) for r in rec['replies'][-2:]]
        if rec['card'] not in KEY:
            continue
        rows.append({'arm': rec['arm'], 'card': rec['card'], 'sample': rec['sample'], **score_run(rec, KEY[rec['card']])})
    (J / 'runs/scores.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    cards = [c for c in KEY if any(r['card'] == c for r in rows)]
    short = {'one': '1', 'one_plus_related': '1+', 'unrelated': 'U', 'cannot_tell': '?', None: '-'}
    print('card'.ljust(8), 'unit/format/truth'.ljust(34), *[a.split('_')[0].ljust(14) for a in ARMS])
    for c in cards:
        k = KEY[c]
        cells = []
        for a in ARMS:
            rs = sorted([r for r in rows if r['arm'] == a and r['card'] == c], key=lambda r: r['sample'])
            if not rs:
                cells.append('.'.ljust(14))
                continue
            cl = ''.join(short.get(r['class'], 'x') for r in rs)
            acc = sum(r['heldout_acc'] or 0 for r in rs) / len(rs)
            cells.append(f'{"/".join(short.get(r["class"], "x") for r in rs)} {acc:.2f}'.ljust(14))
        print(c.ljust(8), f'{k["unit"][:14]}/{k["format"]}/{k["truth_class"]}'.ljust(34), *cells)


if __name__ == '__main__':
    main()
