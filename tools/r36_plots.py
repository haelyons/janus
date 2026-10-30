"""Round 36 figures (presentation only; numbers come from runs/r36/out and runs/r36/metrics_*.json).

1 drift     checkpoint x checkpoint similarity: median over neurons of the Spearman correlation of per-document maxima.
            Lower triangle only. 7B: the sealed sequence. 1B: stage 1, ingredient 3, HF main, post-training.
2 arrival   median (IQR) overlap of each checkpoint's top 16 documents with the final model's, per training phase.
            Final: 7B main (soup), 1B ingredient 3 step 23852 (the amended reference).
3 neurons   8 7B neurons, default_rng(crc32("r36:plots")) out of the 24 on the trajectory page: the rank, at every
            training checkpoint, of each of the final model's top 16 documents.
4 survival  core retention r by the checkpoint where the core formed (groups of at least 5 neurons), null r0 pooled.

usage: python -I tools/r36_plots.py            writes runs/r36/plots/*.png (drift matrices cached as .npy)
"""
import json
import re
import zlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

J = Path(__file__).resolve().parent.parent
R = J / 'runs/r36'
OUT = R / 'plots'
BATCH = {'7B': 4194304, '1B': 2097152}  # tokens per step (step 928646 = 3896B, step 1907359 = 4001B)
SURF, INK, INK2, GRID, NULL = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#8a8984'
COL = {'7B': '#2a78d6', '1B': '#eb6834'}
RAMP = LinearSegmentedColormap.from_list('blue', ['#f3f7fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf',
                                                  '#184f95', '#0d366b'])
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2,
                     'axes.edgecolor': GRID, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 10,
                     'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})


def ckpts(model):
    c = json.loads((R / 'checkpoints.json').read_text())
    if model == '7B':
        return c['allenai/OLMo-2-1124-7B']
    seq = json.loads((R / 'checkpoints_ing3.json').read_text())['allenai/OLMo-2-0425-1B']
    hf = [x for x in c['allenai/OLMo-2-0425-1B'] if x['phase'] == 'main']
    return [x for x in seq if x['phase'] in ('stage1', 'stage2')] + hf + [x for x in seq if x['phase'] not in (
        'stage1', 'stage2')]


def tokens(model, c):
    return c['step'] * BATCH[model]


def short(model, c):
    if c['phase'] == 'stage1':
        t = tokens(model, c)
        return 'init' if t == 0 else f'{t / 1e12:.1f}T' if t >= 1e12 else f'{t / 1e9:.0f}B' if t >= 1e10 else \
            f'{t / 1e9:.1f}B'
    if c['phase'] == 'stage2':
        return f'anneal +{tokens(model, c) / 1e9:.0f}B'
    return {'main': 'soup (final)' if model == '7B' else 'HF main', 'SFT': 'SFT', 'DPO': 'DPO',
            'Instruct': 'Instruct'}[c['phase']]


def neuron_cols(key):
    z = np.load(R / 'out' / f'{key}.npz')
    names = z['units']
    return names, np.array([':n' in n for n in names])


def drift(model):
    cache = OUT / f'drift_{model}.npy'
    cs = ckpts(model)
    if cache.exists():
        return cs, np.load(cache)
    Z = []
    for c in cs:
        z = np.load(R / 'out' / f'{c["key"]}.npz')
        m = np.array([':n' in n for n in z['units']])
        r = rankdata(z['u_max'][:, m], axis=0).astype(np.float32)
        r -= r.mean(0)
        r /= np.maximum(r.std(0), 1e-9)
        Z.append(r)
    K = len(Z)
    C = np.eye(K)
    for a in range(K):
        for b in range(a):
            C[a, b] = C[b, a] = np.median(np.einsum('du,du->u', Z[a], Z[b]) / Z[a].shape[0])
    np.save(cache, C)
    return cs, C


def fig_drift():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), gridspec_kw={'width_ratios': [35, 28], 'wspace': 0.32})
    for ax, model in zip(axes, ('7B', '1B')):
        cs, C = drift(model)
        K = len(cs)
        M = np.where(np.tril(np.ones((K, K), bool), -1), C, np.nan)
        im = ax.imshow(M, cmap=RAMP, vmin=0, vmax=1, interpolation='nearest')
        lab = [short(model, c) for c in cs]
        ax.set_xticks(range(K), lab, rotation=90, fontsize=7)
        ax.set_yticks(range(K), lab, fontsize=7)
        for i, c in enumerate(cs[1:], 1):  # phase boundaries
            if c['phase'] != cs[i - 1]['phase'] and not (c['phase'] in ('DPO', 'Instruct')):
                ax.axhline(i - 0.5, color=SURF, lw=2)
                ax.axvline(i - 0.5, color=SURF, lw=2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f'OLMo-2 {model}', loc='left')
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label('median Spearman ρ over neurons (per-document maxima, 10,000 documents)')
    cb.outline.set_visible(False)
    fig.suptitle("How each neuron's whole response drifts: similarity between every pair of checkpoints", x=0.125,
                 ha='left', fontsize=10, fontweight='bold')
    fig.savefig(OUT / '1_drift.png', dpi=200, bbox_inches='tight')


def metrics(model):
    tag = '' if model == '7B' else '_ing3'
    m = json.loads((R / f'metrics_{model}{tag}.json').read_text())
    S = [c for c in ckpts(model) if c['phase'] in ('stage1', 'stage2') or (model == '7B' and c['phase'] == 'main')]
    assert [c['revision'] for c in S] == m['S']
    return S, [u for u in m['poles']['+']['units'] if ':n' in u['unit']]


def band(v):
    return np.percentile(v, 25), np.median(v), np.percentile(v, 75)


def fig_arrival():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True,
                                     gridspec_kw={'width_ratios': [6, 2.2, 1.6], 'wspace': 0.08})
    for model in ('7B', '1B'):
        S, U = metrics(model)
        Jf = np.array([u['Jf'] for u in U])
        q = np.array([band(Jf[:, t]) for t in range(len(S))])
        st1 = [i for i, c in enumerate(S) if c['phase'] == 'stage1']
        x = np.array([max(tokens(model, S[i]), 1.5e8) for i in st1])
        a1.fill_between(x, q[st1, 0], q[st1, 2], color=COL[model], alpha=0.14, lw=0)
        a1.plot(x, q[st1, 1], color=COL[model], lw=2, marker='o', ms=3, label=f'OLMo-2 {model}')
        a1.annotate(f'{model}  {q[st1[-1], 1]:.2f}', (x[-1], q[st1[-1], 1]), xytext=(-6, 8),
                    textcoords='offset points', ha='right', color=INK, fontsize=8)
        an = [i for i, c in enumerate(S) if c['phase'] == 'stage2' and i != len(S) - 1]
        xa = [tokens(model, S[i]) / 1e9 for i in an]
        a2.fill_between(xa, q[an, 0], q[an, 2], color=COL[model], alpha=0.14, lw=0)
        a2.plot(xa, q[an, 1], color=COL[model], lw=2, marker='o', ms=3)
        off = -0.12 if model == '7B' else 0.12
        for k, ph in enumerate(('SFT', 'DPO', 'Instruct')):
            lo, md, hi = band([u[f'J_{ph}'] for u in U])
            a3.plot([k + off] * 2, [lo, hi], color=COL[model], lw=2, alpha=0.35, solid_capstyle='round')
            a3.plot(k + off, md, 'o', color=COL[model], ms=6)
    a1.set_xscale('log')
    a1.set_xlim(1.2e8, 6e12)
    a1.set_xticks([1.5e8, 1e9, 1e10, 1e11, 1e12], ['init', '1B', '10B', '100B', '1T'])
    a1.set_xlabel('stage 1: pretraining tokens (log)')
    a1.set_ylabel("overlap of top 16 documents with the final model's")
    a1.legend(loc='upper left')
    a2.set_xlim(0, 52)
    a2.set_xlabel('stage 2 anneal: tokens (B)')
    a3.set_xticks(range(3), ['SFT', 'DPO', 'Instruct'])
    a3.set_xlim(-0.5, 2.5)
    a3.set_xlabel('post-training')
    for ax in (a1, a2, a3):
        ax.axhline(0.5, color=GRID, lw=1, zorder=0)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    a1.text(1.3e8, 0.52, 'settled', color=INK2, fontsize=8)
    fig.suptitle('When a neuron\'s final top documents arrive (median and IQR over 1024 / 512 neurons)', x=0.125,
                 ha='left', fontsize=10, fontweight='bold')
    fig.savefig(OUT / '2_arrival.png', dpi=200, bbox_inches='tight')


def fig_neurons():
    S, _ = metrics('7B')
    names, nm = neuron_cols(S[0]['key'])
    neur = np.flatnonzero(nm)
    page = sorted(int(x) for x in np.random.default_rng(zlib.crc32(b'r36:pages')).choice(neur, 24, replace=False))
    page.sort(key=lambda u: (int(names[u].split(':')[0][1:]), names[u]))
    pick = sorted(np.random.default_rng(zlib.crc32(b'r36:plots')).choice(24, 8, replace=False))
    units = [page[i] for i in pick]
    d = np.stack([np.load(R / 'out' / f'{c["key"]}.npz')['u_max'][:, units] for c in S])  # [T, D, 8]
    f = len(S) - 1
    ticks = {i: short('7B', c) for i, c in enumerate(S) if c['phase'] == 'stage1' and c['step'] in (
        10000, 99000, 928646)}
    ticks[f] = 'final'
    a0 = next(i for i, c in enumerate(S) if c['phase'] == 'stage2')
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.6), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.32, 'wspace': 0.08})
    for k, (ax, u) in enumerate(zip(axes.flat, units)):
        rank = np.stack([rankdata(-d[t, :, k], method='min') for t in range(len(S))])  # [T, D]
        top = np.argsort(-d[f, :, k], kind='stable')[:16]
        out64 = rank[:, top] > 64
        entry = np.array([(np.flatnonzero(out64[:, j]).max() + 1) if out64[:, j].any() else 0 for j in range(16)])
        order = np.lexsort((rank[f, top], entry))
        im = ax.imshow(rank[:, top[order]].T, cmap=RAMP.reversed(), norm=LogNorm(1, 10000), aspect='auto',
                       interpolation='nearest')
        ax.axvline(a0 - 0.5, color=SURF, lw=2)
        l, n = names[u].split(':')
        ax.set_title(f'layer {l[1:]} · neuron {int(n[1:]):,}', loc='left', fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks(list(ticks), list(ticks.values()), fontsize=7)
        ax.set_yticks([])
    for ax in axes[:, 0]:
        ax.set_ylabel('final top 16 documents\n(ordered by arrival)', fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015, ticks=[1, 16, 100, 1000, 10000])
    cb.ax.set_yticklabels(['1', '16', '100', '1,000', '10,000'])
    cb.set_label("document's rank at that checkpoint (of 10,000)")
    cb.outline.set_visible(False)
    fig.suptitle('Eight OLMo-2 7B neurons: where their final top documents ranked through training (columns: '
                 '27 stage-1 checkpoints, 4 anneal, final)', x=0.125, ha='left', fontsize=10, fontweight='bold')
    fig.savefig(OUT / '3_neurons.png', dpi=200, bbox_inches='tight')


def fig_survival():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True,
                                 gridspec_kw={'width_ratios': [6, 1], 'wspace': 0.06})
    nulls = []
    for model in ('7B', '1B'):
        S, U = metrics(model)
        cored = [u for u in U if u['t0'] is not None]
        nulls += [u['r0'] for u in cored if u['r0'] is not None]
        groups = {}
        for u in cored:
            groups.setdefault('anneal' if S[u['t0']]['phase'] == 'stage2' else u['t0'], []).append(u['r'])
        for g, v in sorted(groups.items(), key=lambda kv: (kv[0] == 'anneal', kv[0] if kv[0] != 'anneal' else 0)):
            if len(v) < 5:
                continue
            lo, md, hi = band(v)
            ax, x = (a2, 0.15 if model == '1B' else -0.15) if g == 'anneal' else (
                a1, max(tokens(model, S[g]), 1.5e8))
            ax.plot([x, x], [lo, hi], color=COL[model], lw=2, alpha=0.35, solid_capstyle='round')
            ax.scatter([x], [md], s=6 + 0.55 * len(v), color=COL[model], edgecolor=SURF, lw=1,
                       zorder=3)
        a1.plot([], [], 'o', color=COL[model], label=f'OLMo-2 {model}')
    nm = float(np.median(nulls))
    for ax in (a1, a2):
        ax.axhline(nm, color=NULL, lw=1.5, ls=(0, (4, 3)))
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    a1.text(1e10, nm - 0.045, f"null: another neuron's core ({nm:.2f})", color=INK2, fontsize=8)
    for n in (10, 100, 600):
        a1.scatter([], [], s=6 + 0.55 * n, color=INK2, alpha=0.5, label=f'{n} neurons')
    a1.legend(loc='lower right', ncol=2, fontsize=8, columnspacing=1)
    a1.set_xscale('log')
    a1.set_xlim(1.2e8, 6e12)
    a1.set_xticks([1.5e8, 1e9, 1e10, 1e11, 1e12], ['init', '1B', '10B', '100B', '1T'])
    a1.set_xlabel('core formed at: pretraining tokens (log)')
    a1.set_ylabel("core retention: median percentile of the\ncore's documents under the final model")
    a1.set_ylim(0.3, 1.02)
    a2.set_xlim(-0.6, 0.6)
    a2.set_xticks([0], ['anneal'])
    a2.set_xlabel(' ')
    fig.suptitle('Which early cores survive (median and IQR per core checkpoint; groups of 5 or more neurons)', x=0.125,
                 ha='left', fontsize=10, fontweight='bold')
    fig.savefig(OUT / '4_survival.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    fig_arrival()
    fig_survival()
    fig_neurons()
    fig_drift()
    print(sorted(p.name for p in OUT.glob('*.png')))
