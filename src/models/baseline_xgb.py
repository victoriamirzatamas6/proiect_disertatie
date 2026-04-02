import numpy as np
from xgboost import XGBRegressor

def train_xgb(X_train, y_train, params: dict, seed: int):
    model = XGBRegressor(
        n_estimators=int(params.get("n_estimators", 400)),
        max_depth=int(params.get("max_depth", 5)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    return np.asarray(model.predict(X), dtype=float)
