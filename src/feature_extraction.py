import numpy as np

def _features_1d(seg):
    seg = np.asarray(seg)
    # basic TD + slope-based zero crossings with small threshold
    mav = np.mean(np.abs(seg))
    rms = np.sqrt(np.mean(seg**2))
    wl  = np.sum(np.abs(np.diff(seg)))
    # zero crossing with small deadband
    thr = 1e-4
    zc  = np.sum((np.diff(np.sign(seg)) != 0) & (np.abs(seg[:-1] - seg[1:]) > thr))
    return np.array([mav, rms, wl, zc], dtype=float)

def extract_features(signal, window_ms=200, overlap_ms=100, fs=1000):
    """Return features for each window. Supports 1D or 2D (time x channels)."""
    w = int(window_ms * fs / 1000)
    o = int(overlap_ms * fs / 1000)
    step = max(1, w - o)
    if signal.ndim == 1:
        xs = []
        for start in range(0, max(len(signal) - w + 1, 0), step):
            seg = signal[start:start+w]
            xs.append(_features_1d(seg))
        return np.vstack(xs) if len(xs) else np.zeros((0,4))
    else:
        # time x channels
        feats = []
        for start in range(0, max(signal.shape[0] - w + 1, 0), step):
            seg = signal[start:start+w, :]
            # concat features across channels
            fch = [_features_1d(seg[:, c]) for c in range(seg.shape[1])]
            feats.append(np.concatenate(fch, axis=0))
        return np.vstack(feats) if len(feats) else np.zeros((0, 4*signal.shape[1]))
