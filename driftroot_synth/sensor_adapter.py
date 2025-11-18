# sensor_adapter.py – UP-Fall CSV Parser (RAW 36D IMU OUTPUT)

# Driftroot Data Engine v4.0 — Clean, Single-PCA Architecture

import csv
import numpy as np
from sklearn.decomposition import PCA

_GLOBAL_PCA = None

def from_tof(csv_path, sensor_noise=0.01, drift=0.0):
    global _GLOBAL_PCA
    data = []

    # ---- Load CSV ----
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 42:
                continue
            try:
                values = []
                values.extend([float(x) for x in row[1:7]])
                values.extend([float(x) for x in row[8:14]])
                values.extend([float(x) for x in row[15:21]])
                values.extend([float(x) for x in row[22:28]])
                values.extend([float(x) for x in row[29:35]])
                values = values[:36]
                data.append(values)
            except:
                continue

    # ---- EMPTY FILE GUARD ----
    if not data:
        print("  WARNING: Empty or invalid CSV → returning zero-length latent")
        return np.zeros((0, 10))

    X = np.array(data)

    # ---- Inject sensor imperfections ----
    X += np.random.normal(0, sensor_noise, X.shape)
    X += drift

    # ---- Fit PCA once globally ----
    if _GLOBAL_PCA is None:
        if X.shape[0] < 10:
            print("  WARNING: insufficient frames for PCA fit → skipping file")
            return np.zeros((0, 10))

        _GLOBAL_PCA = PCA(n_components=10)
        _GLOBAL_PCA.fit(X)
        print("  Global PCA fitted on 36D → 10D")

    return _GLOBAL_PCA.transform(X)