"""Round 34 checks (PREDICTIONS.md, round 34). Written before any round-34 recording exists.

  python -I tools/r34_check.py [--analysis runs/r34/analysis_suffix.json] [--judge runs/r34/judge_readings.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
MID = range(8, 24)
EARLY = range(0, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', default=str(J / 'runs/r34/analysis_suffix.json'))
    ap.add_argument('--judge', default=str(J / 'runs/r34/judge_readings.json'))
    a = ap.parse_args()
    R = json.loads(Path(a.analysis).read_text())
    lay = R['layers']
    acc = [x['acc_10'] for x in lay]
    mid, early = max(acc[l] for l in MID), max(acc[l] for l in EARLY)
    print(f'P1. best acc_10 in layers 8-23: {mid:.3f} (layer {max(MID, key=lambda l: (acc[l], -l))}) ->', mid >= 0.90)
    print(f'P2. best acc_10 in layers 0-3: {early:.3f}, gap {mid - early:.3f} ->', mid - early >= 0.15)
    sh = [lay[l]['share_top1pct'] for l in MID]
    sg = [lay[l]['share_gauss'] for l in MID]
    print(f'P3. median share of the hidden-space direction in the top 1% of neurons, layers 8-23: {np.median(sh):.3f} '
          f'(random direction {np.median(sg):.3f}) ->', np.median(sh) < 0.5)
    jp = Path(a.judge)
    if jp.exists():
        cats = [x['category'] for x in json.loads(jp.read_text())['judgements']]
        n_ev = sum(c == 'event_meaning' for c in cats)
        print(f'P4. selected neurons whose reading names the event meaning: {n_ev} of {len(cats)} {cats} ->', n_ev <= 2)
    else:
        print('P4. no judge file yet')
    print(f'P5. attention share of the final-layer separation: {R["attn_share"]:.3f} ->', R['attn_share'] >= 0.5)
    ar = max(lay[l]['auc_dir_resid'] for l in MID)
    print(f'P6. best residual-direction AUC in layers 8-23: {ar:.3f} ->', ar >= 0.95)
    print('recorded: per layer acc_1/acc_10/acc_100, auc hidden/resid, share top 1% (gauss):')
    for l, x in enumerate(lay):
        print(f'  L{l:02d} {x["acc_1"]:.2f} {x["acc_10"]:.2f} {x["acc_100"]:.2f}  {x["auc_dir_hidden"]:.2f} '
              f'{x["auc_dir_resid"]:.2f}  {x["share_top1pct"]:.3f} ({x["share_gauss"]:.3f})  eta2 top10 '
              f'{x["eta2_top10"]["meaning"]:.2f}/{x["eta2_top10"]["form"]:.2f}')
    print('recorded: best layer', R['best_layer'], 'selected', R['selected'])
    for f, v in (R.get('factors') or {}).items():
        a10 = v['acc_10']
        print(f'recorded: factor {f} ({v["value"]} vs rest): best acc_10 {max(a10):.3f} at layer {int(np.argmax(a10))}')


if __name__ == '__main__':
    main()
