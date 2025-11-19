# =====================================================================
# Driftroot Synthetic Engine — Version 1.0
# Author: Driftroot Dynamics
# Purpose: ToF CSV ingestion → 36D IMU reconstruction → 10D latent output
# =====================================================================

# Copyright 2025 Driftroot Dynamics LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =====================================================================

import csv
import numpy as np
from sklearn.decomposition import PCA

# Global PCA reused across all CSV files
_GLOBAL_PCA = None


def from_tof(csv_path, sensor_noise=0.01, drift=0.0):
    """
    Loads a ToF CSV with six 6-axis IMU blocks (36D total),
    restores block6 using slice [36:42] (ChatGPT Copilot fix),
    injects sensor imperfections, and transforms to 10D latent.
    """
    global _GLOBAL_PCA
    data = []

    # ---- Load CSV ----
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header

        for row in reader:
            if len(row) < 42:
                continue  # malformed row, skip

            try:
                # Extract six 6-axis blocks (36 total values)
                values = []
                values.extend([float(x) for x in row[1:7]])    # block1
                values.extend([float(x) for x in row[8:14]])   # block2
                values.extend([float(x) for x in row[15:21]])  # block3
                values.extend([float(x) for x in row[22:28]])  # block4
                values.extend([float(x) for x in row[29:35]])  # block5
                values.extend([float(x) for x in row[36:42]])  # block6 (copilot bug fix)

                values = values[:36]  # ensure exactly 36 dims
                data.append(values)

            except Exception:
                continue

    # ---- EMPTY FILE GUARD ----
    if not data:
        print("  WARNING: Empty or invalid CSV → returning zero-length latent")
        return np.zeros((0, 10))

    X = np.array(data, dtype=float)

    # ---- Inject sensor imperfections ----
    X += np.random.normal(0, sensor_noise, X.shape)
    X += drift

    # ---- Fit PCA once globally ----
    if _GLOBAL_PCA is None:
        if X.shape[0] < 10:
            print("  WARNING: insufficient frames for PCA fit → skipping PCA")
            return np.zeros((0, 10))

        _GLOBAL_PCA = PCA(n_components=10)
        _GLOBAL_PCA.fit(X)
        print("  Global PCA fitted on 36D → 10D")

    # ---- Apply PCA transform ----
    latent = _GLOBAL_PCA.transform(X)
    return latent