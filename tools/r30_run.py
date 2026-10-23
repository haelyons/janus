"""Round 30 runner: the round-29 method and the round-20 bare readings, unchanged, on round 30's units, with outputs
redirected under runs/r30<stage>/ by patching module paths at run time (no tool file is edited).

Steps:
- pairtable: tools/pairtable.py main (blind standing answers, pairwise table, readings, checks on T).
- finalread: tools/finalread.py main on that step's stats (round 29B's final reading).
- bare: tools/openmethod.py's own unit_values, make_cards and readings for its arms bare, Q0 and F1, one sample each
  (SAMPLES patched to (1,), WORKERS to 1); no simulation and no scores.
- profile: tools/rangeprofile.py main for --layers (the intervention profile; GPU/CPU only, no Claude calls).
--dry prints, for pairtable and bare, the photos, answer batches and Claude calls a run would make, and makes none.
Phases (so a usage limit cannot cut a long run): answers --part k/n (a slice of pairtable's own answer batches over
the whole union, built as answer_pass builds them), tables --part k/n (run_unit for a slice of units, only once every
batch is cached), bare --part k/n; then the unchanged pairtable step, which finds everything cached, makes no call
and writes the report. A part exits 1 when any of its calls is left uncached; rerunning it redoes only those.

usage: python -I tools/r30_run.py --stage a|b --step pairtable|finalread|bare [--dry]
       python -I tools/r30_run.py --stage a|b --step answers|tables|bare --part k/n
       python -I tools/r30_run.py --stage a --step profile --layers mixed4a
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(J / 'tools'))  # -I does not add the script dir


def units_file(stage):
    return J / f'authors/round30{stage}_units.json'


def redirect_pairtable(out):
    import pairtable
    assert pairtable.answer_pass.__defaults__ == (None, pairtable.OUT, None), pairtable.answer_pass.__defaults__
    assert pairtable.run_unit.__defaults__ == (None, pairtable.OUT), pairtable.run_unit.__defaults__
    pairtable.OUT = out
    pairtable.answer_pass.__defaults__ = (None, out, None)
    pairtable.run_unit.__defaults__ = (None, out)
    return pairtable


def dry_pairtable(uf):
    import proploop
    pairtable = redirect_pairtable(J / 'runs/r30_dry')  # nothing is written there by this function
    units = [proploop.parse_unit(s) for s in json.loads(uf.read_text())]
    names = json.loads(proploop.NAMES.read_text())
    with np.load(proploop.ACTS) as z:
        labels = np.asarray(z['classes'], dtype=np.int64)
    held = {}

    def get_model(dev):  # as pairtable.main
        if dev not in held:
            import bundlemax
            import torch
            torch.set_num_threads(4)
            m = bundlemax._bank().load_model(proploop.WEIGHTS)
            for q in m.parameters():
                q.requires_grad_(False)
            held[dev] = m.to(dev)
        return held[dev]

    layers = sorted({u['layer'] for u in units}, key=proploop.LAYERS.index)
    vals = proploop.layer_values(layers, get_model, np.arange(proploop.N_PHOTOS), out=proploop.OUT)
    held.clear()
    gc.collect()
    per = {u['name']: pairtable.unit_photos(u, vals[u['layer']], labels, names)[1].tolist() for u in units}
    P = sorted({i for v in per.values() for i in v})
    batches = proploop.answer_batches(pairtable.BATCH_KEY, P)
    for n, v in per.items():
        print(f'{n:13s} {len(v)} photos')
    n_read = len(units) * len(pairtable.READINGS)
    print(f'pairtable: {len(P)} photos in the union, {len(batches)} answer calls, {n_read} reading calls '
          f'({len(units)} units x {len(pairtable.READINGS)}), {len(batches) + n_read} in all; finalread up to '
          f'{n_read} more')
    old = J / 'runs/pairtable/answers.json'
    if old.exists():
        seen = {i for r in json.loads(old.read_text()).values() for i in r['photos']}
        print(f'photos also answered in round 29: {len(seen & set(P))}')


def run_pairtable(stage, uf):
    pairtable = redirect_pairtable(J / f'runs/r30{stage}/pairtable')
    sys.argv = ['pairtable.py', '--units-file', str(uf)]
    pairtable.main()


def part_slice(items, part):
    """the k-th of n near-equal consecutive slices of items, part = 'k/n'"""
    k, n = (int(x) for x in part.split('/'))
    assert 1 <= k <= n, part
    b = [round(i * len(items) / n) for i in range(n + 1)]
    return items[b[k - 1]:b[k]]


def prep(stage, uf):
    """as pairtable.main up to the answer pass: units, questions, names, labels, categories, layer values, P"""
    import proploop
    pairtable = redirect_pairtable(J / f'runs/r30{stage}/pairtable')
    proploop.WORKERS = pairtable.WORKERS  # one call at a time
    units = [proploop.parse_unit(s) for s in json.loads(uf.read_text())]
    questions = pairtable.load_questions()
    names = json.loads(proploop.NAMES.read_text())
    with np.load(proploop.ACTS) as z:
        labels = np.asarray(z['classes'], dtype=np.int64)
    from torchvision.models import GoogLeNet_Weights
    cats = list(GoogLeNet_Weights.IMAGENET1K_V1.meta['categories'])
    layers = sorted({u['layer'] for u in units}, key=proploop.LAYERS.index)
    vals = proploop.layer_values(layers, None, np.arange(proploop.N_PHOTOS), out=proploop.OUT)  # cached by the dry run
    P = sorted({i for u in units for i in pairtable.unit_photos(u, vals[u['layer']], labels, names)[1].tolist()})
    pairtable.OUT.mkdir(parents=True, exist_ok=True)
    return pairtable, units, questions, names, labels, cats, vals, P


def answers_part(stage, uf, part):
    """the answer batches of one slice, built exactly as pairtable.answer_pass builds them for the whole union"""
    import contrastloop
    import proploop
    import refineloop
    pairtable, _, questions, _, _, _, _, P = prep(stage, uf)
    batches = proploop.answer_batches(pairtable.BATCH_KEY, P)
    keys = [str(b) for b in range(1, len(batches) + 1)]
    apath = pairtable.OUT / 'answers.json'
    contrastloop.check_answer_cache(apath, batches)
    sel = part_slice(list(zip(keys, batches)), part)
    jobs = [(key, refineloop.d_content(batch, questions, None), proploop._answer_parser(len(batch), questions),
             proploop._answer_fields(batch, len(questions))) for key, batch in sel]
    ac = proploop.cached_calls(pairtable.BATCH_KEY, apath, jobs)
    missing = [k for k, _ in sel if k not in ac]
    print(f'answers part {part}: batches {sel[0][0]}-{sel[-1][0]} of {len(batches)}; not cached {missing}; '
          f'cached in all {sum(k in ac for k in keys)} of {len(keys)}', flush=True)
    return 1 if missing else 0


def tables_part(stage, uf, part):
    """pairtable.run_unit for one slice of units, once every answer batch is cached (no answer call is made here)"""
    import proploop
    pairtable, units, questions, names, labels, cats, vals, P = prep(stage, uf)
    batches = proploop.answer_batches(pairtable.BATCH_KEY, P)
    apath = pairtable.OUT / 'answers.json'
    have = json.loads(apath.read_text()) if apath.exists() else {}
    missing = [str(b) for b in range(1, len(batches) + 1) if str(b) not in have]
    if missing:
        print(f'answer batches not cached: {missing}; run the answers parts first', flush=True)
        return 1
    row_of, _, failed = pairtable.answer_pass(P, questions)
    print(f'answers: {len(P)} photos, batches failed after a retry: {len(failed)} of {len(batches)}', flush=True)
    bad = []
    for u in part_slice(units, part):
        print('==', u['name'], flush=True)
        sec, _ = pairtable.run_unit(u, vals[u['layer']], labels, names, cats, row_of, questions)
        print(sec, flush=True)
        rp = pairtable.OUT / u['name'] / 'readings.json'
        got = json.loads(rp.read_text()) if rp.exists() else {}
        bad += [f'{u["name"]} {k}' for k in pairtable.READINGS if k not in got]
        gc.collect()
    print(f'tables part {part}: readings not cached {bad}', flush=True)
    return 1 if bad else 0


def run_finalread(stage, uf):
    import finalread
    assert finalread.run_unit.__defaults__ == (None, finalread.OUT), finalread.run_unit.__defaults__
    out = J / f'runs/r30{stage}/finalread'
    finalread.OUT, finalread.SRC = out, J / f'runs/r30{stage}/pairtable'
    finalread.run_unit.__defaults__ = (None, out)
    sys.argv = ['finalread.py', '--units-file', str(uf)]
    finalread.main()


def run_bare(stage, uf, dry, part=None):
    import bundlemax
    import openmethod as om
    om.SAMPLES, om.WORKERS = (1,), 1
    units = [om.parse_unit(s) for s in json.loads(uf.read_text())]
    if part:
        units = part_slice(units, part)
    if dry:
        print(f'bare: {len(units)} units x {len(om.ARMS)} arms x {len(om.SAMPLES)} sample = '
              f'{len(units) * len(om.ARMS) * len(om.SAMPLES)} calls, each with 2 photo grids (30 photos)')
        return
    arms = [(arm, '' if f is None else (J / f).read_text().strip()) for arm, f in om.ARMS]
    names = json.loads(om.nearmiss2.NAMES.read_text())
    C, H = bundlemax.pools(names)
    held = []

    def get_model():  # as openmethod.main
        if not held:
            import torch
            torch.set_num_threads(4)
            m = bundlemax._bank().load_model(om.WEIGHTS)
            for p in m.parameters():
                p.requires_grad_(False)
            held.append(m)
        return held[0]

    out = J / f'runs/r30{stage}/bare'
    bad = []
    for u in units:
        print('==', u['name'], flush=True)
        udir = out / u['name']
        udir.mkdir(parents=True, exist_ok=True)
        d = om.unit_values(u, C, H, udir, get_model)
        meta, content = om.make_cards(u, d, udir)
        om.nearmiss2.save_json(meta, udir / 'card.json')
        recs = om.readings(u, content, arms, udir)
        for k in sorted(recs):
            print(f'  {k}: {str(recs[k].get("explanation"))[:160]}', flush=True)
        bad += [f'{u["name"]} {a}_s{k}' for a, _ in arms for k in om.SAMPLES if f'{a}_s{k}' not in recs]
        gc.collect()
    print(f'bare part {part}: readings not cached {bad}', flush=True)
    return 1 if bad else 0


def run_profile(stage, layers):
    import rangeprofile
    out = J / f'runs/r30{stage}/rangeprofile'
    assert rangeprofile.load_profiles.__defaults__[1] == rangeprofile.OUT, rangeprofile.load_profiles.__defaults__
    rangeprofile.OUT = out
    rangeprofile.load_profiles.__defaults__ = (rangeprofile.load_profiles.__defaults__[0], out)
    uf = out / 'units.json'
    out.mkdir(parents=True, exist_ok=True)
    sel = [u for u in json.loads(units_file(stage).read_text()) if u.split(':')[0] in layers.split(',')]
    uf.write_text(json.dumps(sel) + '\n')
    sys.argv = ['rangeprofile.py', '--layers', layers, '--units-file', str(uf)]
    rangeprofile.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['a', 'b'])
    ap.add_argument('--step', required=True,
                    choices=['pairtable', 'answers', 'tables', 'finalread', 'bare', 'profile'])
    ap.add_argument('--part', help="k/n: the k-th of n slices (answers: answer batches; tables, bare: units)")
    ap.add_argument('--layers', default='mixed4a')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    uf = units_file(a.stage)
    if a.step == 'pairtable':
        return dry_pairtable(uf) if a.dry else run_pairtable(a.stage, uf)
    if a.step == 'answers':
        return answers_part(a.stage, uf, a.part)
    if a.step == 'tables':
        return tables_part(a.stage, uf, a.part)
    if a.step == 'finalread':
        return run_finalread(a.stage, uf)
    if a.step == 'bare':
        return run_bare(a.stage, uf, a.dry, a.part)
    return run_profile(a.stage, a.layers)


if __name__ == '__main__':
    sys.exit(main())
