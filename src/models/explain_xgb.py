import numpy as np
from sklearn.inspection import permutation_importance

def xgb_gain_importance(model):
    return np.asarray(model.feature_importances_, dtype=float)

def permutation_importance_reg(model, X, y, seed: int = 42, n_repeats: int = 10):
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return np.asarray(r.importances_mean, dtype=float)
