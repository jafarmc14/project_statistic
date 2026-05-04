import os, json, yaml, datetime, random
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import torch

from src.load_uci import load_uci_dataset
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.models import build_models, evaluate_models
from src.evaluation import subjectwise_split, print_report
from src.visualization import plot_bar, plot_confusion

from src.eda import (
    plot_class_distribution,
    plot_raw_signal_example,
    plot_preprocessed_signal_example,
    plot_windowing_example,
    plot_feature_correlation_heatmap,
    plot_feature_boxplots,
    plot_pca_2d
)

def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    return v


def _safe_name(s):
    return "".join(c if c.isalnum() or c in "-_()" else "_" for c in str(s))


def _seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device(cfg):
    use_gpu = cfg.get("use_gpu", True)
    gpu_id = int(cfg.get("gpu_id", 0))
    if use_gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def _aggregate_results(run_results_by_model):
    summary = {}
    for model_name, runs in run_results_by_model.items():
        accs = np.array([r["accuracy"] for r in runs], dtype=float)
        f1s = np.array([r["macro_f1"] for r in runs], dtype=float)

        summary[model_name] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "accuracy_min": float(np.min(accs)),
            "accuracy_max": float(np.max(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_sd": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "macro_f1_min": float(np.min(f1s)),
            "macro_f1_max": float(np.max(f1s)),
            "n_runs": int(len(runs))
        }
    return summary

def _per_class_from_report(report_dict, class_to_id):
    """
    Ubah classification_report(output_dict=True) menjadi ringkasan per kelas
    dengan nama kelas asli.
    """
    id_to_class = {str(v): str(k) for k, v in class_to_id.items()}
    out = {}

    for cls_id_str, cls_name in id_to_class.items():
        if cls_id_str not in report_dict:
            continue
        out[cls_name] = {
            "precision": float(report_dict[cls_id_str]["precision"]),
            "recall": float(report_dict[cls_id_str]["recall"]),
            "f1_score": float(report_dict[cls_id_str]["f1-score"]),
            "support": int(report_dict[cls_id_str]["support"])
        }

    return out

def _feature_names():
    chans = ["RF", "BF", "VM", "ST"]
    feats = ["MAV", "RMS", "WL", "ZC"]
    return [f"{ch}_{ft}" for ch in chans for ft in feats]

def run(cfg_path="D:/emg-baseline/emg-baseline/configs/uci_baseline.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    base_seed = int(cfg.get("random_state", 42))
    _seed_everything(base_seed)
    device = _get_device(cfg)

    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU name: {torch.cuda.get_device_name(device)}")

    df = load_uci_dataset(cfg["data_path"])

    plot_class_distribution(df, out_path="figures/eda_class_distribution.png")

    # contoh satu file mentah
    plot_raw_signal_example(df, sample_idx=0, out_path="figures/eda_raw_signal_example.png")

    print(df[["file", "subject_id", "label"]].head())

    for i in range(min(5, len(df))):
        print(df.iloc[i]["file"], np.asarray(df.iloc[i]["signal"]).shape)

    fs = cfg.get("sampling_rate", 1000)
    win = cfg.get("window", 200)
    ovl = cfg.get("overlap", 100)
    n_repeats = int(cfg.get("n_repeats", 10))
    save_all_runs = bool(cfg.get("save_all_runs", True))

    # preprocess sekali saja
    X_feat_list, y_list, g_list = [], [], []
    saved_example_signals = False
    for sig, label, subject_id in tqdm(
        zip(df["signal"], df["label"], df["subject_id"]),
        total=len(df),
        desc="Preprocessing + feature extraction"
    ):
        x = preprocess(sig, fs=fs)

        if not saved_example_signals:
            plot_preprocessed_signal_example(
                raw_signal=sig,
                proc_signal=x,
                label=label,
                subject_id=subject_id,
                out_path="figures/eda_preprocessed_signal_example.png"
            )
            plot_windowing_example(
                proc_signal=x,
                fs=fs,
                window_ms=win,
                overlap_ms=ovl,
                out_path="figures/eda_windowing_example.png"
            )
            saved_example_signals = True

        feats = extract_features(x, window_ms=win, overlap_ms=ovl, fs=fs)

        if feats.shape[0] == 0:
            continue

        y = np.array([label] * feats.shape[0])
        g = np.array([subject_id] * feats.shape[0])

        X_feat_list.append(feats)
        y_list.append(y)
        g_list.append(g)

    if len(X_feat_list) == 0:
        raise RuntimeError("No feature windows produced. Coba ubah window/overlap di configs.")

    X = np.vstack(X_feat_list).astype(np.float32)
    y = np.concatenate(y_list)
    groups = np.concatenate(g_list)
    feature_names = _feature_names()[:X.shape[1]]
    plot_feature_correlation_heatmap(X, out_path="figures/eda_feature_correlation_heatmap.png")
    plot_feature_boxplots(X, y, out_path="figures/eda_feature_boxplots.png")
    plot_pca_2d(X, y, out_path="figures/eda_pca_2d.png")

    classes_sorted = sorted(np.unique(y))
    class_to_id = {c: i for i, c in enumerate(classes_sorted)}
    y_enc = np.array([class_to_id[v] for v in y], dtype=np.int64)

    run_results_by_model = defaultdict(list)
    run_details = []
    best_run_overall = None

    dataset_name = cfg.get("dataset", "UCI Lower Limb EMG")

    for repeat_idx in range(n_repeats):
        seed = base_seed + repeat_idx
        print(f"\n{'='*20} Repeat {repeat_idx+1}/{n_repeats} | seed={seed} {'='*20}")
        _seed_everything(seed)

        X_train, X_test, y_train, y_test, g_train, g_test = subjectwise_split(
            X, y_enc, groups,
            test_size=cfg.get("test_size", 0.2),
            random_state=seed
        )

        train_subjects = sorted(set(g_train.tolist()))
        test_subjects = sorted(set(g_test.tolist()))
        overlap_subjects = sorted(set(train_subjects).intersection(set(test_subjects)))

        if overlap_subjects:
            raise RuntimeError(f"Subject leakage detected: {overlap_subjects}")

        print(f"[INFO] Train subjects: {len(train_subjects)} | Test subjects: {len(test_subjects)}")
        print(f"[INFO] Train/Test subject overlap: {len(overlap_subjects)}")

        models = build_models(
            cfg.get("models", ["SVM", "RF", "MLP"]),
            input_dim=X_train.shape[1],
            n_classes=len(classes_sorted),
            device=device,
            random_state=seed,
            mlp_params=cfg.get("mlp_params", {})
        )

        results, best_name, best_pred = evaluate_models(
            models,
            X_train, y_train,
            X_test, y_test,
            device=device
        )

        repeat_record = {
            "repeat_index": repeat_idx + 1,
            "seed": seed,
            "test_size": cfg.get("test_size", 0.2),
            "results_per_model": {}
        }

        repeat_record["split_info"] = {
            "n_train_subjects": int(len(train_subjects)),
            "n_test_subjects": int(len(test_subjects)),
            "train_subjects": list(train_subjects),
            "test_subjects": list(test_subjects)
        }

        for model_name, metrics in results.items():
            pred = None
            # recompute predictions per model for confusion if model is best in this repeat only not necessary
            run_results_by_model[model_name].append({
                "repeat_index": repeat_idx + 1,
                "seed": seed,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"])
            })
            repeat_record["results_per_model"][model_name] = {
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"])
            }

        # simpan run terbaik global berdasarkan macro-F1 terbaik tunggal
        best_acc = float(accuracy_score(y_test, best_pred))
        best_f1 = float(f1_score(y_test, best_pred, average="macro"))
        best_cm = confusion_matrix(y_test, best_pred).tolist()
        best_rep = classification_report(y_test, best_pred, output_dict=True)
        best_rep_per_class = _per_class_from_report(best_rep, class_to_id)

        repeat_record["best_model_this_repeat"] = {
            "model": best_name,
            "accuracy": best_acc,
            "macro_f1": best_f1,
            "confusion_matrix": best_cm,
            "classification_report": best_rep,
            "per_class_metrics": best_rep_per_class
        }

        if (best_run_overall is None) or (best_f1 > best_run_overall["macro_f1"]):
            best_run_overall = {
                "repeat_index": repeat_idx + 1,
                "seed": seed,
                "model": best_name,
                "accuracy": best_acc,
                "macro_f1": best_f1,
                "confusion_matrix": best_cm,
                "classification_report": best_rep,
                "y_test": y_test.tolist(),
                "best_pred": best_pred.tolist(),
                "per_class_metrics": best_rep_per_class,
            }

        run_details.append(repeat_record)

    aggregated = _aggregate_results(run_results_by_model)

    # pilih model terbaik berdasarkan mean macro-F1
    best_model_mean = max(aggregated.items(), key=lambda kv: kv[1]["macro_f1_mean"])[0]

    # bar plot: langsung pakai aggregated agar SD ikut tampil sebagai error bars
    plot_bar(aggregated, f"{dataset_name} (Repeated Hold-out)")

    plot_confusion(
        np.array(best_run_overall["y_test"]),
        np.array(best_run_overall["best_pred"]),
        f"{dataset_name} (Best Repeat)"
    )

    fig_bar = f"figures/bar_{_safe_name(dataset_name + ' (Repeated Hold-out)')}.png"
    fig_conf = f"figures/confusion_{_safe_name(dataset_name + ' (Best Repeat)')}.png"

    print("\n=== Aggregated Summary (mean ± SD) ===")
    for model_name, stats in aggregated.items():
        print(
            f"{model_name}: "
            f"acc={stats['accuracy_mean']:.4f} ± {stats['accuracy_sd']:.4f}, "
            f"macro_f1={stats['macro_f1_mean']:.4f} ± {stats['macro_f1_sd']:.4f}"
        )

    print(f"\nBest model by mean macro-F1: {best_model_mean}")

    print("\n=== Per-class metrics (best single run) ===")
    for cls_name, m in best_run_overall["per_class_metrics"].items():
        print(
            f"{cls_name}: "
            f"precision={m['precision']:.4f}, "
            f"recall={m['recall']:.4f}, "
            f"f1={m['f1_score']:.4f}, "
            f"support={m['support']}"
        )

    class_dist = {str(k): int(v) for k, v in Counter(y_enc).items()}

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"uci_repeated_holdout_({win}ms)_{n_repeats}x"
    out_json = os.path.join("logs", f"{_safe_name(base)}_{stamp}.json")

    payload = {
        "created_at": stamp,
        "dataset": dataset_name,
        "mode": "baseline_repeated_holdout",
        "split_strategy": "subject-wise GroupShuffleSplit",
        "n_unique_subjects": int(len(np.unique(groups))),
        "device": str(device),
        "config": {
            "cfg_path": cfg_path,
            "data_path": cfg.get("data_path"),
            "sampling_rate": fs,
            "window_ms": win,
            "overlap_ms": ovl,
            "test_size": cfg.get("test_size", 0.2),
            "random_state_base": base_seed,
            "n_repeats": n_repeats,
            "models": cfg.get("models", ["SVM", "RF", "MLP"]),
            "use_gpu": cfg.get("use_gpu", True),
            "gpu_id": cfg.get("gpu_id", 0),
            "mlp_params": cfg.get("mlp_params", {})
        },
        "data_shape": {
            "n_windows": int(X.shape[0]),
            "features_per_window": int(X.shape[1]),
            "n_classes": int(len(classes_sorted))
        },
        "feature_names": feature_names,
        "class_map": {str(int(class_to_id[c])): _safe_name(c) for c in classes_sorted},
        "class_dist": class_dist,
        "aggregated_results_per_model": aggregated,
        "best_model_by_mean_macro_f1": best_model_mean,
        "best_single_run": {
            "repeat_index": best_run_overall["repeat_index"],
            "seed": best_run_overall["seed"],
            "model": best_run_overall["model"],
            "accuracy": best_run_overall["accuracy"],
            "macro_f1": best_run_overall["macro_f1"],
            "confusion_matrix": best_run_overall["confusion_matrix"],
            "classification_report": best_run_overall["classification_report"],
            "per_class_metrics": best_run_overall["per_class_metrics"],
        },
        "figures": {
            "bar": fig_bar,
            "confusion": fig_conf
        }
    }

    if save_all_runs:
        payload["all_runs"] = run_details

    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=_py)

    print(f"[LOG] Repeated hold-out summary saved to {out_json}")


if __name__ == "__main__":
    run()