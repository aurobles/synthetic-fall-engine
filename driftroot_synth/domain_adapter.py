# =====================================================================
# Driftroot Synthetic Engine — Version 1.0
# Author: Driftroot Dynamics
# Purpose: Generate personalized, physics-based synthetic fall data
# =====================================================================

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