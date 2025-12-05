# =====================================================================
# Synthetic Fall Engine — Version 1.0
# Author: Aurobles
# Purpose: Generate personalized, physics-based synthetic fall data
# =====================================================================

import numpy as np
import random

class FederatedSynthesizer:

    def __init__(self):
        pass

    # ----------------------------------------------
    # RESIDENT PERSONALIZATION
    # ----------------------------------------------
    def personalize_by_resident(self, seq, profile=None):
        """
        Profile example:
        {
            "age": 86,
            "mobility": "walker" | "cane" | "independent"
        }
        """
        seq = seq.copy()
        if not profile:
            return seq

        age = profile.get("age", None)
        mobility = profile.get("mobility", None)

        # Age-based amplitude scaling
        if age is not None:
            seq *= 1 + ((age - 70) * 0.002)

        # Mobility-based micro-jitter
        if mobility == "walker":
            seq += np.random.normal(0, 0.02, seq.shape)
        elif mobility == "cane":
            seq += np.random.normal(0, 0.01, seq.shape)

        return seq

    # ----------------------------------------------
    # ROOM PERSONALIZATION
    # ----------------------------------------------
    def personalize_by_room(self, seq, room_profile=None):
        """
        room_profile example:
        {
            "type": "double" | "memory_care" | "single",
            "layout_noise": float
        }
        """
        seq = seq.copy()
        if not room_profile:
            return seq

        room_type = room_profile.get("type", None)
        layout_noise = room_profile.get("layout_noise", 0.01)

        if room_type == "double":
            seq += np.random.normal(0, 0.015, seq.shape)
        elif room_type == "memory_care":
            seq += np.random.normal(0, 0.03, seq.shape)

        # Generic environment noise
        seq += np.random.normal(0, layout_noise, seq.shape)

        return seq

    # ----------------------------------------------
    # SENSOR DEVICE PROFILE (Device→Device variance)
    # ----------------------------------------------
    def apply_sensor_profile(self, seq, sensor_profile=None):
        """
        sensor_profile example:
        {
            "noise": float,
            "drift": float
        }
        """
        seq = seq.copy()
        if not sensor_profile:
            return seq

        noise = sensor_profile.get("noise", 0.01)
        drift = sensor_profile.get("drift", 0.0)

        seq += np.random.normal(0, noise, seq.shape)
        seq += drift

        return seq

    # ----------------------------------------------
    # MASTER PERSONALIZATION PIPELINE
    # ----------------------------------------------
    def personalize(self, seq, resident=None, room=None, sensor=None):
        seq = self.personalize_by_resident(seq, resident)
        seq = self.personalize_by_room(seq, room)
        seq = self.apply_sensor_profile(seq, sensor)
        return seq
