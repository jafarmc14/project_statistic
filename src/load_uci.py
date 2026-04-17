import os
import numpy as np
import pandas as pd

import re

_float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _load_txt_file(path, assume_first_is_time=True, prefer_last_channel=True):
    """
    Robust loader for SEMG_DB1 txt files:
    - extracts all floats per line via regex
    - keeps only rows that match the most common column count
    - if multi-column: optionally drop first column (time) and
      then either take last channel or mean across channels
    Returns: 1D numpy array (signal)
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            # extract floats anywhere in the line
            nums = _float_re.findall(s)
            if not nums:
                continue
            try:
                vals = [float(x) for x in nums]
            except Exception:
                continue
            rows.append(vals)

    if not rows:
        raise RuntimeError(f"No numeric lines found in {path}")

    # Determine most common column count (mode)
    lens = [len(r) for r in rows]
    # mode robustly
    from collections import Counter
    L, _ = Counter(lens).most_common(1)[0]

    # Keep only rows with that column count
    rows = [r for r in rows if len(r) == L]
    if len(rows) == 0:
        raise RuntimeError(f"No consistent rows with {L} columns in {path}")

    data = np.asarray(rows, dtype=float)  # shape (n, L)

    # If multi-column, select signal columns
    if data.ndim == 2 and data.shape[1] > 1:
        start_col = 1 if assume_first_is_time else 0
        sig_part = data[:, start_col:]
        if sig_part.shape[1] == 1:
            signal = sig_part[:, 0]
        else:
            if prefer_last_channel:
                signal = sig_part[:, -1]      # gunakan kanal terakhir
            else:
                signal = sig_part.mean(axis=1)  # atau rata-rata semua kanal
    else:
        signal = data.squeeze()

    # Jaga-jaga: buang NaN/Inf
    mask = np.isfinite(signal)
    signal = signal[mask]
    if signal.size == 0:
        raise RuntimeError(f"All values non-finite in {path}")

    return signal

def load_uci_dataset(base_path):
    """Load UCI Lower-Limb EMG (SEMG_DB1) with A_TXT (abnormal) and N_TXT (normal).
    Returns DataFrame with columns: signal (1D array), label (str), file (str).
    """
    data, labels, files = [], [], []
    n_dir = os.path.join(base_path, "N_TXT")
    a_dir = os.path.join(base_path, "A_TXT")
    if not os.path.isdir(n_dir) or not os.path.isdir(a_dir):
        raise FileNotFoundError("Expected N_TXT and A_TXT folders inside the UCI dataset path.")

    for f in sorted(os.listdir(n_dir)):
        if f.lower().endswith(".txt"):
            p = os.path.join(n_dir, f)
            sig = _load_txt_file(p)
            data.append(sig); labels.append("normal"); files.append(p)

    for f in sorted(os.listdir(a_dir)):
        if f.lower().endswith(".txt"):
            p = os.path.join(a_dir, f)
            sig = _load_txt_file(p)
            data.append(sig); labels.append("abnormal"); files.append(p)

    df = pd.DataFrame({"signal": data, "label": labels, "file": files})
    return df
