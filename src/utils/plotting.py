from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def save_pred_plot(df, x_col, y_true_col, y_pred_col, title, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(x_col)
    plt.figure()
    plt.plot(df[x_col].values, df[y_true_col].values, label="True")
    plt.plot(df[x_col].values, df[y_pred_col].values, label="Pred")
    plt.xlabel(x_col); plt.ylabel(y_true_col)
    plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def save_error_hist(y_true, y_pred, title, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    err = np.asarray(y_pred) - np.asarray(y_true)
    plt.figure()
    plt.hist(err, bins=50)
    plt.xlabel("Prediction error (pred - true)"); plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def save_score_hist(train_scores, test_scores, title, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(train_scores, bins=50, alpha=0.7, label="train")
    plt.hist(test_scores, bins=50, alpha=0.7, label="test")
    plt.xlabel("score"); plt.ylabel("count")
    plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def save_feature_importance(names, values, title, out_path, top_n=15):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    idx = np.argsort(values)[::-1][:top_n]
    plt.figure(figsize=(8, 5))
    plt.barh([names[i] for i in idx][::-1], [values[i] for i in idx][::-1])
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
