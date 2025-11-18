# =============================================================================
# kinesthetic_mapper.py – PCA-Based Kinesthetic Reducer for Driftroot Synth
# Reduces 54-dim ToF point clouds to 10-dim latent vectors
# Features: Fit-once transform-many | 2018 ToF dataset trained
# UP-Fall Compatible | ESP32 TinyML Ready
# =============================================================================
#
# Copyright 2025 Driftroot Dynamics LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE:
# This module is currently unused in pipeline v3.3.
# Retained for future PCA research, multi-sensor alignment,
# and model explainability development.

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