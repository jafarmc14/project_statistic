import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class TorchMLPNet(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dims=(128, 64, 32), dropout=0.2):
        super().__init__()

        dims = [input_dim] + list(hidden_dims)
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLPClassifier:
    """
    Scikit-like wrapper untuk PyTorch MLP agar tetap kompatibel dengan evaluate_models().

    Improvement:
    - menambahkan StandardScaler internal
    - scaler di-fit hanya pada train set
    - X_val dan X_test ditransform menggunakan scaler yang sama
    """
    def __init__(
        self,
        input_dim,
        n_classes,
        device="cpu",
        hidden_dims=(128, 64, 32),
        dropout=0.2,
        lr=1e-3,
        batch_size=512,
        epochs=60,
        weight_decay=1e-4,
        patience=8,
        random_state=42,
        val_fraction=0.1
    ):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.device = torch.device(device)
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.random_state = random_state
        self.val_fraction = val_fraction

        # scaler khusus untuk MLP
        self.scaler = StandardScaler()

        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        self.model = TorchMLPNet(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # standardization fit hanya pada data train fold ini
        X = self.scaler.fit_transform(X).astype(np.float32)

        n = len(X)
        idx = np.arange(n)
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(idx)

        val_size = max(1, int(self.val_fraction * n))
        val_idx = idx[:val_size]
        tr_idx = idx[val_size:]

        X_tr = X[tr_idx]
        y_tr = y[tr_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        # class weights untuk imbalance
        classes = np.unique(y_tr)
        cls_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        full_class_weights = np.ones(self.n_classes, dtype=np.float32)
        for c, w in zip(classes, cls_weights):
            full_class_weights[int(c)] = float(w)

        class_weights_tensor = torch.tensor(full_class_weights, dtype=torch.float32, device=self.device)

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=(self.device.type == "cuda")
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t.to(self.device))
                val_loss = criterion(val_logits, y_val_t.to(self.device)).item()

            print(
                f"    [MLP] Epoch {epoch+1:03d}/{self.epochs} | "
                f"train_loss={running_loss / max(1, len(X_tr)):.4f} | val_loss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"    [MLP] Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        X = self.scaler.transform(X).astype(np.float32)
        self.model.eval()

        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(xb)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()

        return pred


def build_models(names, input_dim=None, n_classes=None, device="cpu", random_state=42, mlp_params=None):
    models = {}
    mlp_params = mlp_params or {}

    if "SVM" in names:
        models["SVM"] = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced"
            )
        )

    if "RF" in names:
        models["RF"] = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state
        )

    if "MLP" in names:
        if input_dim is None or n_classes is None:
            raise ValueError("MLP requires input_dim and n_classes.")

        models["MLP"] = TorchMLPClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            device=device,
            random_state=random_state,
            **mlp_params
        )

    return models


def evaluate_models(models, X_train, y_train, X_test, y_test, device="cpu"):
    """
    Evaluasi semua model:
    - SVM: StandardScaler + SVC
    - RF : tanpa scaling
    - MLP: scaler internal + PyTorch

    Return:
    - results: metrik per model
    - best_name: model terbaik pada repeat ini
    - best_pred: prediksi model terbaik pada repeat ini
    - preds_by_model: prediksi semua model
    """
    results = {}
    preds_by_model = {}
    best_name, best_f1, best_pred = None, -1.0, None

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            preds_by_model[name] = pred

            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="macro", zero_division=0)
            results[name] = {"accuracy": float(acc), "macro_f1": float(f1)}

            print(f"{name} → Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1, best_name, best_pred = f1, name, pred

        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            continue

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"{k}: acc={v['accuracy']:.4f}, macro_f1={v['macro_f1']:.4f}")
    print(f"\nBest model: {best_name}")

    return results, best_name, best_pred, preds_by_model