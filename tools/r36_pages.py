"""Round 36 trajectory page: 24 seeded 7B neurons, top 3 contexts at every checkpoint (presentation only).

Neurons: default_rng(crc32("r36:pages")).choice(the 7B drawn neurons, 24, replace=False). Per checkpoint (training
sequence, then SFT, DPO, Instruct): J with main, and the 3 documents with the highest per-document max, each shown as
12 ids before and after the argmax position, the argmax token marked. Settling and core come from
runs/r36/metrics_7B.json (tools/r36_analyze.py). Writes runs/r36/pages/trajectories.html.

usage: python -I tools/r36_pages.py
"""
import json
import re
import zlib
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
R = J / 'runs/r36'
SIDE, SHOW, N = 12, 3, 24


def label(c):
    if c['phase'] == 'stage1':
        tok = re.search(r'tokens(\d+)B', c['revision']).group(1)
        return f"step {c['step']:,}", f'{tok}B'
    if c['phase'] == 'stage2':
        tok = re.search(r'tokens(\d+)B', c['revision']).group(1)
        return f"anneal step {c['step']:,}", f'+{tok}B'
    return {'main': ('main (final base)', ''), 'SFT': ('SFT', ''), 'DPO': ('DPO', ''),
            'Instruct': ('Instruct (RLVR)', '')}[c['phase']]


def main():
    import sys
    sys.path.insert(0, str(J / 'tools'))
    import r36_analyze as A
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(R / 'tok7B'))
    P = dict(np.load(R / 'pool_7B.npz'))
    names, S, post, d, pos = A.load('7B', '+')
    cks = S + list(post.values())
    met = {u['unit']: u for u in json.loads((R / 'metrics_7B.json').read_text())['poles']['+']['units']}
    neur = [i for i, n in enumerate(names) if ':n' in n]
    pick = sorted(int(x) for x in np.random.default_rng(zlib.crc32(b'r36:pages')).choice(neur, N, replace=False))
    pick.sort(key=lambda u: (int(names[u].split(':')[0][1:]), names[u]))
    f = len(S) - 1
    out = []
    for u in pick:
        m = met[names[u]]
        Tf = set(A.topk(d[f, :, u], 16).tolist())
        rows = []
        for t, c in enumerate(cks):
            top = A.topk(d[t, :, u], SHOW)
            ctx = []
            for x in top:
                ids = P['ids'][P['off'][x]:P['off'][x + 1]].tolist()
                p = int(pos[t, x, u])
                a, b = max(1, p - SIDE), min(len(ids), p + SIDE + 1)
                ctx.append({'l': tok.decode(ids[a:p]), 'm': tok.decode(ids[p:p + 1]), 'r': tok.decode(ids[p + 1:b]),
                            'v': round(float(d[t, x, u]), 3), 'fin': int(x) in Tf})
            jf = len(set(A.topk(d[t, :, u], 16).tolist()) & Tf) / 16
            step, tk = label(c)
            rows.append({'step': step, 'tok': tk, 'phase': c['phase'], 'J': jf, 'ctx': ctx})
        l, n = names[u].split(':')
        out.append({'id': f'{l}-{n}', 'layer': int(l[1:]), 'index': int(n[1:]), 's': m['s'], 't0': m['t0'],
                    'r': m['r'], 'r0': m['r0'], 'o': m['o'], 'rows': rows})
    page = (J / 'tools/r36_pages_template.html').read_text().replace('/*DATA*/null', json.dumps(out))
    (R / 'pages').mkdir(exist_ok=True)
    (R / 'pages/trajectories.html').write_text(page)
    print(R / 'pages/trajectories.html', len(page) // 1024, 'KB')


if __name__ == '__main__':
    main()
