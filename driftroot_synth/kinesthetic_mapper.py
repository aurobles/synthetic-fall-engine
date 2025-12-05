# =====================================================================
# Synthetic Fall Engine — Version 1.0
# Author: Aurobles
# Purpose: Generate personalized, physics-based synthetic fall data
# =====================================================================

import numpy as np
from sklearn.decomposition import PCA


class KinestheticMapper:
    def __init__(self, n_components=10, input_dim=36):
        """
        Create a PCA reducer for kinesthetic latent space.

        Parameters:
            n_components: output dimension (usually 10)
            input_dim: expected input dimension (36 for UP-Fall IMU)
        """
        self.n_components = n_components
        self.input_dim = input_dim
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, X):
        """
        Fit PCA to the first N frames of a real trial.
        X must be shaped (T, 36).
        """
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"KinestheticMapper expected input_dim={self.input_dim} "
                f"but got {X.shape[1]}"
            )

        self.pca.fit(X)
        self.is_fitted = True

    def transform(self, frame):
        """
        Transform a single 36D frame → 10D latent vector.
        """
        frame = np.asarray(frame)

        if frame.shape[-1] != self.input_dim:
            raise ValueError(
                f"Frame must be {self.input_dim} dims but got {frame.shape[-1]}"
            )

        if not self.is_fitted:
            raise RuntimeError(
                "KinestheticMapper.transform() called before PCA was fitted."
            )

        # Reshape to (1, 36), transform, flatten → (10,)
        return self.pca.transform(frame.reshape(1, -1))[0]
