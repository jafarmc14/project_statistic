import os
import numpy as np
import pandas as pd
import re

_float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _load_txt_file(path):
    """
    Load SEMG_DB1 txt file and always return EMG-only data
    with shape (time, 4), corresponding to RF, BF, VM, ST.

    Supported layouts per row:
    - 4 cols  : RF BF VM ST
    - 5 cols  : RF BF VM ST FX
    - 6 cols  : time RF BF VM ST FX
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue

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

    from collections import Counter
    lens = [len(r) for r in rows]
    L, _ = Counter(lens).most_common(1)[0]

    rows = [r for r in rows if len(r) == L]
    if len(rows) == 0:
        raise RuntimeError(f"No consistent rows with {L} columns in {path}")

    data = np.asarray(rows, dtype=float)

    # Normalisasi layout agar output selalu (n_samples, 4)
    if L == 4:
        # langsung 4 channel EMG
        emg = data[:, 0:4]

    elif L == 5:
        # kemungkinan RF BF VM ST FX
        emg = data[:, 0:4]

    elif L >= 6:
        # kemungkinan time + RF BF VM ST FX
        emg = data[:, 1:5]

    else:
        raise RuntimeError(f"Unexpected column count {L} in {path}")

    # buang baris yang non-finite
    mask = np.all(np.isfinite(emg), axis=1)
    emg = emg[mask]

    if emg.size == 0:
        raise RuntimeError(f"All EMG values non-finite in {path}")
    
    if emg.shape[1] != 4:
        raise RuntimeError(f"Expected 4 EMG channels, got {emg.shape[1]} in {path}")

    return emg

def _parse_filename(fname):
    """
    Contoh:
      10Amar.txt -> subject_num=10, cohort=A, activity_code=mar
      7Npie.txt  -> subject_num=7, cohort=N, activity_code=pie
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r"^(\d+)([AN])([A-Za-z]+)$", base, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unexpected filename format: {fname}")

    subject_num = int(m.group(1))
    cohort_code = m.group(2).upper()
    activity_code = m.group(3).lower()

    cohort = "abnormal" if cohort_code == "A" else "normal"

    # Berdasarkan pola nama file + deskripsi aktivitas
    # pie ≈ standing (de pie), sen ≈ sitting (sentado), mar ≈ walking (marcha)
    activity_map = {
        "mar": "walking",
        "pie": "standing",
        "sen": "sitting"
    }
    if activity_code not in activity_map:
        raise ValueError(f"Unknown activity code '{activity_code}' in {fname}")

    activity = activity_map[activity_code]

    # subject global unik, jangan hanya 1..11
    subject_id = f"{cohort_code}{subject_num:02d}"

    return subject_id, subject_num, cohort, activity, activity_code

def load_uci_dataset(base_path):
    """
    Returns DataFrame with columns:
    signal, label, subject_id, subject_num, cohort, activity_code, file
    """
    records = []

    for subdir, cohort_code in [("N_TXT", "N"), ("A_TXT", "A")]:
        d = os.path.join(base_path, subdir)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing folder: {d}")

        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".txt"):
                continue

            p = os.path.join(d, f)
            signal = _load_txt_file(p)

            subject_id, subject_num, cohort, activity, activity_code = _parse_filename(f)

            records.append({
                "signal": signal,              # 2D: time x channels
                "label": activity,            # target klasifikasi aktivitas
                "subject_id": subject_id,     # mis. A01, N03
                "subject_num": subject_num,   # 1..11
                "cohort": cohort,             # normal / abnormal
                "activity_code": activity_code,
                "file": p
            })

    df = pd.DataFrame(records)
    return df