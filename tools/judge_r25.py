"""Round 25 judge: round 24's references, on runs/proploop/readings_all.json.

usage: python -I tools/judge_r25.py   (writes runs/proploop/judge.json)
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
import judge_r24  # noqa: E402,F401  (sets judge_r20.REF, 443 and 429 included)
import judge_r20  # noqa: E402

WORKERS = 3


def main():
    rd = json.loads((J / 'runs/proploop/readings_all.json').read_text())
    jobs = [(u, rid, t) for u in rd if u in judge_r20.REF for rid, t in rd[u].items() if t]
    with ThreadPoolExecutor(WORKERS) as ex:
        got = list(ex.map(lambda j: judge_r20.ask(*j), jobs))
    res = {}
    for (u, rid, _), g in zip(jobs, got):
        res.setdefault(u, {})[rid] = g
    (J / 'runs/proploop/judge.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for u in res:
        print(u, ' '.join(f'{rid}={g["match"]}{"D" if g["names_direction"] else ""}' for rid, g in res[u].items()))


if __name__ == '__main__':
    main()
