import numpy as np
from sklearn.decomposition import PCA

def fit_pca_reconstruction(X_train: np.ndarray, n_components: int = 10, seed: int = 42):
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_train)
    return pca

def reconstruction_error(pca: PCA, X: np.ndarray) -> np.ndarray:
    X_proj = pca.transform(X)
    X_hat = pca.inverse_transform(X_proj)
    return np.mean((X - X_hat) ** 2, axis=1).astype(float)
