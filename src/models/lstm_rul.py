import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)

def make_sequences(df, feature_cols, window: int, cap_rul: int | None, step: int = 1):
    X_list, y_list, meta = [], [], []
    for unit_id, g in df.groupby("unit_id"):
        g = g.sort_values("cycle")
        vals = g[feature_cols].values
        rul  = g["RUL"].values
        cyc  = g["cycle"].values
        for end in range(window, len(g) + 1, step):
            X_list.append(vals[end-window:end, :])
            y = float(rul[end-1])
            if cap_rul is not None:
                y = float(min(y, cap_rul))
            y_list.append(y)
            meta.append((int(unit_id), int(cyc[end-1])))
    return np.stack(X_list), np.asarray(y_list, dtype=float), meta

def train_lstm(X_tr, y_tr, X_va, y_va, cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMRegressor(X_tr.shape[-1], int(cfg["hidden_size"]), int(cfg["num_layers"]), float(cfg["dropout"])).to(device)
    tr = DataLoader(SeqDataset(X_tr, y_tr), batch_size=int(cfg["batch_size"]), shuffle=True)
    va = DataLoader(SeqDataset(X_va, y_va), batch_size=int(cfg["batch_size"]), shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    patience = int(cfg["patience"])
    for _ in range(int(cfg["epochs"])):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                vals.append(loss_fn(model(xb), yb).item())
        cur = float(np.mean(vals))
        if cur < best:
            best = cur
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_lstm(model, X, batch_size: int = 512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device); model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out).astype(float)
