"""Peer baseline for one channel's response to an image edit, InceptionV1 mixed4e.

For an edit x -> y and a unit u: a_c is the spatial max of channel c's post-ReLU 14x14 map, computed by
tools/battery.acts. s_c, the strong level, is the median of a_c over c's top-20 pool photos, from battery's caches
runs/battery/{pool,orig,topk}.npy (loaded once and kept). Peers of x are the channels c != u with
a_c(x) >= 0.5 * s_c (and a_c(x) > 0, so that r_c is defined). r_c = a_c(y) / a_c(x) over peers,
unit_ratio = a_u(y) / a_u(x) (NaN if a_u(x) <= 0), unit_pct = 100 x fraction of peers with r_c <= unit_ratio.

usage: python -I tools/peers.py --selftest
       peer_baseline(load_model(), [(x, y), ...], u) -> [{'n', 'peer_median', 'unit_ratio', 'unit_pct'}, ...]
"""
import sys
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parent.parent
if str(J / 'tools') not in sys.path:
    sys.path.insert(0, str(J / 'tools'))
import battery  # noqa: E402  tools/battery.py, reused not copied

_held = {}


def baseline_from_acts(ax, ay, s, unit):
    """Peers of x for unit u and where u's a(y) / a(x) falls among theirs. ax = a(x), ay = a(y), s = strong levels."""
    ax, ay, s = (np.asarray(v, dtype=np.float64) for v in (ax, ay, s))
    assert ax.shape == ay.shape == s.shape == (battery.NCH,), (ax.shape, ay.shape, s.shape)
    u = int(unit)
    assert 0 <= u < battery.NCH, u
    peer = (ax >= 0.5 * s) & (ax > 0)
    peer[u] = False
    n = int(peer.sum())
    r = battery.ratio(ay[peer], ax[peer])
    ur = float(battery.ratio(ay[u], ax[u]))
    return {
        'n': n,
        'peer_median': float(np.median(r)) if n else float('nan'),
        'unit_ratio': ur,
        'unit_pct': 100 * float(np.mean(r <= ur)) if n and not np.isnan(ur) else float('nan'),
    }


def strong_levels(P, orig, topk):
    """s_c = median of a_c over c's top-20 photos. P pool ids, orig (len(P), 832) a(x) on P, topk (832, 20) ids."""
    P = np.asarray(P)
    assert orig.shape == (len(P), battery.NCH), orig.shape
    assert topk.shape == (battery.NCH, battery.K), topk.shape
    assert (np.diff(P) > 0).all(), 'pool ids not strictly increasing'
    pos = np.searchsorted(P, topk)
    assert (pos < len(P)).all() and np.array_equal(P[pos], topk), 'topk ids not in pool'
    return np.median(np.take_along_axis(orig.T, pos, axis=1), axis=1).astype(np.float64)


def strong():
    """Strong levels from runs/battery caches, loaded on first call and kept, (832,) float64."""
    if 's' not in _held:
        o = battery.OUT
        _held['s'] = strong_levels(np.load(o / 'pool.npy'), np.load(o / 'orig.npy'), np.load(o / 'topk.npy'))
    return _held['s']


def _acts(model, imgs):
    """a(img) for PIL RGB 224x224 images through battery.acts, (len(imgs), 832) float32."""
    for im in imgs:
        assert im.mode == 'RGB' and im.size == (224, 224), (im.mode, im.size)
    photo = battery.photo
    battery.photo = lambda i: imgs[int(i)]  # battery.acts loads photo ids from disk; hand it these images instead
    try:
        return battery.acts(model, np.arange(len(imgs)))
    finally:
        battery.photo = photo


def peer_baseline(model, pairs, unit):
    """baseline_from_acts for each (x, y) in pairs; all 2 * len(pairs) images in one battery.acts pass (batches of 64)."""
    if not pairs:
        return []
    s = strong()
    a = _acts(model, [im for x, y in pairs for im in (x, y)])
    return [baseline_from_acts(a[2 * i], a[2 * i + 1], s, unit) for i in range(len(pairs))]


def load_model():
    """InceptionV1 from raw/weights/inception5h.pth, loaded as tools/battery.py does (4 torch threads)."""
    import torch
    torch.set_num_threads(4)
    return battery._bank().load_model(J / 'raw/weights/inception5h.pth')


def decide(d):
    """Neutral label on unit_pct: <= 10 BELOW_PEERS, >= 90 ABOVE_PEERS, else WITHIN_PEERS; UNDEFINED if NaN."""
    p = d['unit_pct']
    if np.isnan(p):
        return 'UNDEFINED'
    return 'BELOW_PEERS' if p <= 10 else 'ABOVE_PEERS' if p >= 90 else 'WITHIN_PEERS'


def selftest():
    nan = float('nan')
    u = 7

    def check(d, want, what):
        assert list(d) == ['n', 'peer_median', 'unit_ratio', 'unit_pct'], (what, list(d))
        assert isinstance(d['n'], int) and d['n'] == want[0], (what, d)
        for k, w in zip(['peer_median', 'unit_ratio', 'unit_pct'], want[1:]):
            v = d[k]
            assert (np.isnan(v) and np.isnan(w)) or abs(v - w) <= 1e-9, (what, k, v, w)

    def synth():
        s = np.full(battery.NCH, 2.0)
        ax, ay = np.full(battery.NCH, 0.5), np.zeros(battery.NCH)  # a_c(x) = 0.5 < 0.5 * s_c: not peers
        ax[u], ay[u] = 4.0, 2.0  # unit, itself above threshold; unit_ratio 0.5
        for c, (x, y) in {10: (1.0, 0.25), 11: (3.0, 1.5), 12: (2.0, 2.0), 13: (5.0, 0.0), 14: (1.5, 3.0)}.items():
            ax[c], ay[c] = x, y  # peers, c = 10 exactly at 0.5 * s_c; r = 0.25, 0.5, 1.0, 0.0, 2.0
        ax[20], ay[20] = np.nextafter(1.0, 0.0), 9.0  # just below 0.5 * s_c: not a peer
        s[30], ax[30], ay[30] = 0.0, 0.0, 9.0  # a_c(x) = 0.5 * s_c = 0, r_c undefined: not a peer
        return ax, ay, s

    ax, ay, s = synth()
    check(baseline_from_acts(ax, ay, s, u), (5, 0.5, 0.5, 60.0), 'inclusive threshold, unit excluded, tie <=')
    check(baseline_from_acts(ax, ay, s, 10), (5, 0.5, 0.25, 20.0), 'unit 10 excluded, channel 7 now a peer')
    ay[u] = 0.0
    check(baseline_from_acts(ax, ay, s, u), (5, 0.5, 0.0, 20.0), 'tie at 0 counted <=')
    ay[u] = 40.0
    check(baseline_from_acts(ax, ay, s, u), (5, 0.5, 10.0, 100.0), 'above every peer')
    ax[u] = 0.0
    check(baseline_from_acts(ax, ay, s, u), (5, 0.5, nan, nan), 'a_u(x) = 0')

    ax, ay, s = synth()
    ax[14] = 0.5
    check(baseline_from_acts(ax, ay, s, u), (4, 0.375, 0.5, 75.0), 'even peer count median')
    ax[10:15] = 0.5
    check(baseline_from_acts(ax, ay, s, u), (0, nan, 0.5, nan), 'no peers')
    ax[u] = 0.0
    check(baseline_from_acts(ax, ay, s, u), (0, nan, nan, nan), 'no peers, a_u(x) = 0')

    for p, lab in [(nan, 'UNDEFINED'), (0.0, 'BELOW_PEERS'), (10.0, 'BELOW_PEERS'), (50.0, 'WITHIN_PEERS'),
                   (90.0, 'ABOVE_PEERS'), (100.0, 'ABOVE_PEERS')]:
        assert decide({'unit_pct': p}) == lab, (p, decide({'unit_pct': p}), lab)

    rng = np.random.default_rng(12345)
    P = np.sort(rng.choice(5000, 300, replace=False))
    orig = rng.random((len(P), battery.NCH), dtype=np.float32)
    topk = P[np.argsort(-orig, axis=0, kind='stable')[:battery.K].T]  # as battery.run builds topk.npy
    want = np.median(np.sort(orig, axis=0)[-battery.K:], axis=0).astype(np.float64)
    assert np.array_equal(strong_levels(P, orig, topk), want), 'strong levels'
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    print(__doc__)


if __name__ == '__main__':
    main()
