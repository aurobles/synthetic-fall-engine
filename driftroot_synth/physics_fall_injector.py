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
import random
from scipy.integrate import odeint

DIRECTIONS = ["forward", "backward", "left", "right"]

def inject_fall(seq, direction=None, severity=1.0):
    seq = seq.copy()
    T = len(seq)

    if T < 20:
        return seq, "adl"

    direction = direction or random.choice(DIRECTIONS)
    fall_start = random.randint(T // 4, 3 * T // 4)

    # Time steps (~30Hz)
    t = np.linspace(0, (T - fall_start) * 0.033, T - fall_start)

    # State vector:
    # θ = pitch angle
    # dθ = angular velocity
    # x,y = center-of-mass displacement
    # vx,vy = linear velocities
    initial = [
        0.05 * severity,     # θ
        0.0,                 # dθ
        0.0, 1.7,            # x, y height
        0.0, 0.0             # vx, vy
    ]

    def physics(state, t):
        θ, dθ, x, y, vx, vy = state

        g = 9.81
        L = 1.0              # pendulum length
        damping = 0.35

        # Angular acceleration (inverted pendulum)
        ddθ = (g / L) * np.sin(θ) - damping * dθ

        # Linear motion
        ax = L * ddθ
        ay = -g + (-damping * vy)

        return [dθ, ddθ, vx, vy, ax, ay]

    traj = odeint(physics, initial, t)

    # Map motion into 6 latent dimensions
    seq[fall_start:, :6] = np.clip(traj[:, :6], -50, 50)

    # Add impact spike
    impact = fall_start + int(8 * severity)
    if impact < T:
        seq[impact, 2:4] += 55 * severity

    return seq, f"fall_{direction}"