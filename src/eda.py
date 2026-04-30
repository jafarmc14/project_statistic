import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def _ensure_dir(path="figures"):
    os.makedirs(path, exist_ok=True)

def _feature_names():
    chans = ["RF", "BF", "VM", "ST"]
    feats = ["MAV", "RMS", "WL", "ZC"]
    return [f"{ch}_{ft}" for ch in chans for ft in feats]

def plot_class_distribution(df, out_path="figures/eda_class_distribution.png"):
    _ensure_dir(os.path.dirname(out_path) or ".")
    counts = df["label"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(counts.index.astype(str), counts.values)

    plt.xlabel("Class")
    plt.ylabel("Number of recordings")
    plt.title("Class distribution")

    # tambahkan angka di atas bar
    for bar, val in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.ylim(0, max(counts.values) + 2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_raw_signal_example(df, sample_idx=0, out_path="figures/eda_raw_signal_example.png"):
    _ensure_dir(os.path.dirname(out_path) or ".")
    sig = np.asarray(df.iloc[sample_idx]["signal"])
    label = df.iloc[sample_idx]["label"]
    subject_id = df.iloc[sample_idx].get("subject_id", "unknown")

    ch_names = ["RF", "BF", "VM", "ST"]
    n_ch = sig.shape[1] if sig.ndim == 2 else 1

    plt.figure(figsize=(10, 6))
    if sig.ndim == 1:
        plt.plot(sig, linewidth=0.8)
        plt.title(f"Raw EMG signal | subject={subject_id} | class={label}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
    else:
        for c in range(n_ch):
            plt.plot(sig[:, c], linewidth=0.8, label=ch_names[c] if c < len(ch_names) else f"Ch{c+1}")
        plt.title(f"Raw EMG signal | subject={subject_id} | class={label}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_preprocessed_signal_example(raw_signal, proc_signal, label, subject_id, out_path="figures/eda_preprocessed_signal_example.png"):
    _ensure_dir(os.path.dirname(out_path) or ".")
    ch_names = ["RF", "BF", "VM", "ST"]

    plt.figure(figsize=(12, 8))

    if raw_signal.ndim == 1:
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(raw_signal, linewidth=0.8)
        ax1.set_title(f"Raw signal | subject={subject_id} | class={label}")
        ax1.set_xlabel("Samples")
        ax1.set_ylabel("Amplitude")

        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(proc_signal, linewidth=0.8)
        ax2.set_title("Preprocessed signal")
        ax2.set_xlabel("Samples")
        ax2.set_ylabel("Amplitude")
    else:
        ax1 = plt.subplot(2, 1, 1)
        for c in range(raw_signal.shape[1]):
            ax1.plot(raw_signal[:, c], linewidth=0.8, label=ch_names[c] if c < len(ch_names) else f"Ch{c+1}")
        ax1.set_title(f"Raw signal | subject={subject_id} | class={label}")
        ax1.set_xlabel("Samples")
        ax1.set_ylabel("Amplitude")
        ax1.legend()

        ax2 = plt.subplot(2, 1, 2)
        for c in range(proc_signal.shape[1]):
            ax2.plot(proc_signal[:, c], linewidth=0.8, label=ch_names[c] if c < len(ch_names) else f"Ch{c+1}")
        ax2.set_title("Preprocessed signal")
        ax2.set_xlabel("Samples")
        ax2.set_ylabel("Amplitude")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_windowing_example(proc_signal, fs=1000, window_ms=200, overlap_ms=100,
                           out_path="figures/eda_windowing_example.png"):
    _ensure_dir(os.path.dirname(out_path) or ".")
    w = int(window_ms * fs / 1000)
    o = int(overlap_ms * fs / 1000)
    step = max(1, w - o)

    if proc_signal.ndim == 2:
        y = proc_signal[:, 0]  # tampilkan channel pertama saja
    else:
        y = proc_signal

    plt.figure(figsize=(12, 4))
    plt.plot(y, linewidth=0.8)

    ymax = np.max(y) if len(y) else 1.0

    # tampilkan hanya 3 window pertama
    max_windows_to_draw = 3
    starts = list(range(0, max(len(y) - w + 1, 0), step))[:max_windows_to_draw]

    for i, s in enumerate(starts):
        e = s + w
        plt.axvspan(s, e, alpha=0.2)

    # posisi label dibuat manual agar tidak bertumpuk
    label_x_positions = [
        starts[0] + w * 0.18,
        starts[1] + w * 0.50,
        starts[2] + w * 0.82
    ]

    for i, x_pos in enumerate(label_x_positions):
        plt.text(
            x_pos,
            ymax * 0.97,
            f"W{i+1}",
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
        )

    plt.title(f"Windowing example ({window_ms} ms window, {overlap_ms} ms overlap)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def plot_feature_correlation_heatmap(X, out_path="figures/eda_feature_correlation_heatmap.png"):
    _ensure_dir(os.path.dirname(out_path) or ".")
    feat_names = _feature_names()
    n_features = X.shape[1]
    feat_names = feat_names[:n_features]

    df_feat = pd.DataFrame(X, columns=feat_names)
    corr = df_feat.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(feat_names)), feat_names, rotation=90)
    plt.yticks(range(len(feat_names)), feat_names)
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_boxplots(X, y_labels, out_path="figures/eda_feature_boxplots.png",
                          selected_features=None):
    _ensure_dir(os.path.dirname(out_path) or ".")

    feat_names = _feature_names()[:X.shape[1]]

    # default fitur yang lebih representatif
    if selected_features is None:
        selected_features = ["RF_MAV", "RF_WL", "VM_RMS"]

    # cek apakah fitur tersedia
    selected_idx = []
    selected_names = []
    for feat in selected_features:
        if feat in feat_names:
            selected_idx.append(feat_names.index(feat))
            selected_names.append(feat)

    unique_classes = sorted(pd.unique(y_labels))

    fig, axes = plt.subplots(len(selected_idx), 1, figsize=(8, 3.5 * len(selected_idx)))

    if len(selected_idx) == 1:
        axes = [axes]

    for ax, feat_idx, feat_name in zip(axes, selected_idx, selected_names):
        data_by_class = [X[np.array(y_labels) == cls, feat_idx] for cls in unique_classes]
        ax.boxplot(data_by_class, labels=unique_classes)
        ax.set_title(f"Boxplot of {feat_name} by class")
        ax.set_xlabel("Class")
        ax.set_ylabel(feat_name)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_pca_2d(X, y_labels, out_path="figures/eda_pca_2d.png", max_points=4000):
    _ensure_dir(os.path.dirname(out_path) or ".")
    X = np.asarray(X)
    y_labels = np.asarray(y_labels)

    if len(X) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_points, replace=False)
        X_plot = X[idx]
        y_plot = y_labels[idx]
    else:
        X_plot = X
        y_plot = y_labels

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_plot)

    classes = sorted(np.unique(y_plot))

    plt.figure(figsize=(7, 6))
    for cls in classes:
        mask = y_plot == cls
        plt.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.6, label=str(cls))

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection of extracted features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()