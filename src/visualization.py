import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


_illegal = re.compile(r'[\\/:*?"<>|]')


def _safe_name(title: str) -> str:
    name = title.replace(" ", "_")
    name = _illegal.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def plot_bar(results, title, outdir="figures"):
    """
    Support 2 formats:
    1) single-run:
       {
         "SVM": {"accuracy": 0.73, "macro_f1": 0.70},
         "RF":  {"accuracy": 0.91, "macro_f1": 0.90}
       }

    2) repeated hold-out aggregated:
       {
         "SVM": {
             "accuracy_mean": 0.73, "accuracy_sd": 0.01,
             "macro_f1_mean": 0.70, "macro_f1_sd": 0.01
         },
         "RF": {
             "accuracy_mean": 0.91, "accuracy_sd": 0.00,
             "macro_f1_mean": 0.90, "macro_f1_sd": 0.00
         }
       }
    """
    ensure_dir(outdir)

    names = list(results.keys())
    first = results[names[0]]

    is_repeated = ("accuracy_mean" in first) or ("macro_f1_mean" in first)

    if is_repeated:
        acc_vals = [results[k]["accuracy_mean"] for k in names]
        acc_errs = [results[k].get("accuracy_sd", 0.0) for k in names]
        f1_vals  = [results[k]["macro_f1_mean"] for k in names]
        f1_errs  = [results[k].get("macro_f1_sd", 0.0) for k in names]
        plot_title = f"Model Comparison – {title}\n(Mean ± SD across repeated hold-out runs)"
    else:
        acc_vals = [results[k]["accuracy"] for k in names]
        acc_errs = [0.0 for _ in names]
        f1_vals  = [results[k]["macro_f1"] for k in names]
        f1_errs  = [0.0 for _ in names]
        plot_title = f"Model Comparison – {title}"

    x = np.arange(len(names))
    width = 0.36

    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(x - width/2, acc_vals, width, yerr=acc_errs, capsize=5, label="Accuracy")
    bars2 = plt.bar(x + width/2, f1_vals, width, yerr=f1_errs, capsize=5, label="Macro F1")

    plt.xticks(x, names)
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title(plot_title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bars, vals in [(bars1, acc_vals), (bars2, f1_vals)]:
        for bar, val in zip(bars, vals):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    safe = _safe_name(title)
    path = os.path.join(outdir, f"bar_{safe}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path

def plot_loso_boxplot(summary, title, metric="f1", ylabel="Macro F1", outdir="figures"):
    """
    summary format:
    {
      "SVM": {"subjects":[...], "acc":[...], "f1":[...]},
      "RF": {...},
      "MLP": {...}
    }
    """
    ensure_dir(outdir)

    names = list(summary.keys())
    data = [summary[k][metric] for k in names]

    plt.figure(figsize=(7, 5))
    bp = plt.boxplot(data, labels=names, patch_artist=True, showmeans=True)

    colors = ["#d9e6f2", "#dff0d8", "#fce5cd"]
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)

    plt.ylabel(ylabel)
    plt.ylim(0, 1.0)
    plt.title(f"{title}\n(LOSO per-subject distribution)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for i, vals in enumerate(data, start=1):
        vals = np.asarray(vals, dtype=float)
        med = np.median(vals)
        plt.text(i, med + 0.02, f"{med:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    safe = _safe_name(title)
    path = os.path.join(outdir, f"boxplot_{safe}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_loso_mean_sd(summary, title, metric="f1", ylabel="Macro F1", outdir="figures"):
    """
    summary format:
    {
      "SVM": {"subjects":[...], "acc":[...], "f1":[...]},
      ...
    }
    """
    ensure_dir(outdir)

    names = list(summary.keys())
    vals = [np.asarray(summary[k][metric], dtype=float) for k in names]
    means = [float(v.mean()) for v in vals]
    sds = [float(v.std(ddof=1)) if len(v) > 1 else 0.0 for v in vals]

    x = np.arange(len(names))

    plt.figure(figsize=(7, 5))
    bars = plt.bar(x, means, yerr=sds, capsize=5)

    plt.xticks(x, names)
    plt.ylabel(ylabel)
    plt.ylim(0, 1.0)
    plt.title(f"{title}\n(Mean ± SD across LOSO subjects)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    safe = _safe_name(title)
    path = os.path.join(outdir, f"loso_mean_sd_{safe}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path

def plot_confusion(y_true, y_pred, title, outdir="figures"):
    ensure_dir(outdir)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix – {title}")
    plt.colorbar()

    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")

    plt.tight_layout()
    safe = _safe_name(title)
    path = os.path.join(outdir, f"confusion_{safe}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path