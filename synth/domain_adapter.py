# =====================================================================
# Synthetic Fall Engine — Version 1.0
# Author: Aurobles
# Purpose: Generate personalized, physics-based synthetic fall data
# =====================================================================

import numpy as np

class DomainAdapter:

    def mmd(self, X, Y):
        """Maximum Mean Discrepancy (very simplified)."""
        return abs(X.mean() - Y.mean()) + abs(X.std() - Y.std())

    def coral(self, X, Y):
        """CORAL distance (covariance alignment)."""
        Cx = np.cov(X.T)
        Cy = np.cov(Y.T)
        return np.linalg.norm(Cx - Cy)

    def real_to_synthetic_gap(self, real, synth):
        return {
            "mmd": float(self.mmd(real, synth)),
            "coral": float(self.coral(real, synth))
        }
