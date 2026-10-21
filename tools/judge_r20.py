"""Round 20 judge: a blind call rates each reading of a unit against a reference description (the literature answer
for ground-truth units, the essay's or Zoom In's claim for the others). 3 calls per reading, median taken.

usage: python -I tools/judge_r20.py   (reads runs/openmethod/readings_all.json, writes runs/openmethod/judge.json)
"""
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
from claude_call import converse  # noqa: E402

DOG = 'dog heads seen facing one particular direction (turned to the left, or turned to the right)'
REF = {'mixed4b:237': 'car windows', 'mixed4b:491': 'car bodies (the body panels of a car)', 'mixed4b:373': 'car wheels',
       'mixed4b:443': DOG, 'mixed4b:431': DOG, 'mixed4c:254': DOG, 'mixed4c:407': DOG,
       'mixed5a:719': 'a pinched-center shape, wide at both ends and narrow in the middle, as in bow ties, butterflies and '
                      'swimsuit tops',
       'mixed4d:417': 'a pose or view shared by cars and dogs', 'mixed4d:241': 'a pose or view shared by cars and dogs',
       'mixed4d:277': 'a pose or view shared by cars and dogs'}
PROMPT = ('Here is a reference description of what one unit (neuron) of an image classification network responds to, '
          'and a candidate explanation of the same unit written by someone else.\n\nReference: {ref}\n\nCandidate: {cand}\n\n'
          'Does the candidate identify what the reference describes as what the unit responds to? Rate 1 (no, it describes '
          'something else) to 5 (yes, the same thing, as the main description). Also say whether the candidate names a '
          'direction the subject faces or is turned (left, right, or one particular side). Reply with only JSON: '
          '{{"match": n, "names_direction": true or false}}')
CALLS = 3


def ask(u, rid, text):
    out = []
    for _ in range(CALLS):
        r = converse('', [[{'type': 'text', 'text': PROMPT.format(ref=REF[u], cand=text)}]])[0]['text']
        try:
            o = json.loads(r[r.index('{'):r.rindex('}') + 1])
            out.append({'match': int(o['match']), 'names_direction': bool(o['names_direction']), 'text': r})
        except (ValueError, KeyError, TypeError):
            out.append({'match': None, 'names_direction': None, 'text': r})
    ok = [o for o in out if o['match'] is not None]
    return {'calls': out, 'match': statistics.median(o['match'] for o in ok) if ok else None,
            'names_direction': (sum(o['names_direction'] for o in ok) * 2 > len(ok)) if ok else None}


def main():
    rd = json.loads((J / 'runs/openmethod/readings_all.json').read_text())
    jobs = [(u, rid, t) for u in rd if u in REF for rid, t in rd[u].items() if t]
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda j: ask(*j), jobs))
    res = {}
    for (u, rid, _), g in zip(jobs, got):
        res.setdefault(u, {})[rid] = g
    (J / 'runs/openmethod/judge.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for u in res:
        print(u, ' '.join(f'{rid}={g["match"]}{"D" if g["names_direction"] else ""}' for rid, g in res[u].items()))


if __name__ == '__main__':
    main()
