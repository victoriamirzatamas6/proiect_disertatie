import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        x = self.X[idx]
        return x, x

class AE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
    def forward(self, x): return self.decoder(self.encoder(x))

def make_flat_windows(df: pd.DataFrame, feature_cols: list[str], window: int):
    X_list, meta = [], []
    for uid, g in df.groupby("unit_id"):
        g = g.sort_values("cycle")
        vals = g[feature_cols].values
        cyc = g["cycle"].values
        for end in range(window, len(g)+1):
            X_list.append(vals[end-window:end, :].reshape(-1))
            meta.append((int(uid), int(cyc[end-1])))
    return np.stack(X_list), meta

def _filter_normal_part(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    parts = []
    for uid, g in df.groupby("unit_id"):
        g = g.sort_values("cycle")
        cut = max(1, int(len(g)*ratio))
        parts.append(g.iloc[:cut])
    return pd.concat(parts).copy()

def train_autoencoder(X_train: np.ndarray, cfg: dict, X_val: np.ndarray | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AE(X_train.shape[1], int(cfg["hidden_dim"]), int(cfg["latent_dim"])).to(device)
    dl = DataLoader(WindowDataset(X_train), batch_size=int(cfg["batch_size"]), shuffle=True)
    val_dl = DataLoader(WindowDataset(X_val), batch_size=int(cfg["batch_size"]), shuffle=False) if X_val is not None and len(X_val) > 0 else None
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    patience = int(cfg["patience"])
    for _ in range(int(cfg["epochs"])):
        model.train()
        tr_losses = []
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        if val_dl is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    val_losses.append(loss_fn(model(xb), yb).item())
            cur = float(np.mean(val_losses))
        else:
            cur = float(np.mean(tr_losses))
        if cur < best:
            best = cur
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def reconstruction_error(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device); model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            xh = model(xb)
            errs.append(torch.mean((xh - xb) ** 2, dim=1).cpu().numpy())
    return np.concatenate(errs).astype(float)

def fit_and_score(df_train: pd.DataFrame, df_test: pd.DataFrame, feature_cols: list[str], cfg: dict):
    window = int(cfg["window_size"])
    df_normal = _filter_normal_part(df_train, float(cfg["normal_cycles_ratio"])) if bool(cfg.get("train_normal_only", False)) else df_train
    # Split normal data by unit (80/20) for validation-based early stopping
    units = np.array(sorted(df_normal["unit_id"].unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(units)
    n_tr = max(1, int(len(units) * 0.8))
    tr_units = set(units[:n_tr].tolist())
    va_units = set(units[n_tr:].tolist())
    df_tr_split = df_normal[df_normal["unit_id"].isin(tr_units)]
    df_va_split = df_normal[df_normal["unit_id"].isin(va_units)]
    X_tr, _ = make_flat_windows(df_tr_split, feature_cols, window)
    X_va, _ = make_flat_windows(df_va_split, feature_cols, window) if not df_va_split.empty else (None, None)
    X_val_arg = X_va if X_va is not None and len(X_va) > 0 else None
    model = train_autoencoder(X_tr, cfg, X_val=X_val_arg)
    # Calculate threshold on all normal training data for a representative estimate
    X_normal_all, _ = make_flat_windows(df_normal, feature_cols, window)
    train_scores = reconstruction_error(model, X_normal_all, batch_size=int(cfg["batch_size"]))
    X_te, meta_te = make_flat_windows(df_test, feature_cols, window)
    test_scores = reconstruction_error(model, X_te, batch_size=int(cfg["batch_size"]))
    return model, train_scores, test_scores, meta_te
