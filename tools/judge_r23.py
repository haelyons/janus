"""Round 23 judge: round 21's references (tools/judge_r21.py) plus 4b:443 with 4b:431's, on
runs/sparedview/readings_all.json; 6 calls in parallel.

usage: python -I tools/judge_r23.py   (writes runs/sparedview/judge.json)
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))
import judge_r21  # noqa: E402,F401  (sets judge_r20.REF)
import judge_r20  # noqa: E402

judge_r20.REF['mixed4b:443'] = judge_r20.REF['mixed4b:431']


def main():
    rd = json.loads((J / 'runs/sparedview/readings_all.json').read_text())
    jobs = [(u, rid, t) for u in rd if u in judge_r20.REF for rid, t in rd[u].items() if t]
    with ThreadPoolExecutor(6) as ex:
        got = list(ex.map(lambda j: judge_r20.ask(*j), jobs))
    res = {}
    for (u, rid, _), g in zip(jobs, got):
        res.setdefault(u, {})[rid] = g
    (J / 'runs/sparedview/judge.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for u in res:
        print(u, ' '.join(f'{rid}={g["match"]}{"D" if g["names_direction"] else ""}' for rid, g in res[u].items()))


if __name__ == '__main__':
    main()
