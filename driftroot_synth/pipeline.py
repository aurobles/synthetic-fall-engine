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

import os
import glob
import json
import random
import numpy as np
from tqdm import tqdm   

from sensor_adapter import from_tof
from physics_fall_injector import inject_fall
from federated_synthesizer import FederatedSynthesizer
from domain_adapter import DomainAdapter


# =====================================================================
# GLOBAL REPRODUCIBILITY SEED 
# =====================================================================

np.random.seed(42)
random.seed(42)


# =====================================================================
# CONFIGURATION
# =====================================================================

DEMO_MODE = False               # Full mode ON
SAMPLES_PER_TRIAL = 100 if not DEMO_MODE else 10

# CHANGE THIS to your local path before running
dataset_path = "path/to/UP-Fall-Dataset/Subject1"
output_dir   = r"path/to/driftroot-synth/driftroot_synth/data/simulations"

os.makedirs(output_dir, exist_ok=True)


# =====================================================================
# VALIDATION
# =====================================================================

def validate_generation(seq, label):
    """Quick sanity check for quality."""
    if np.any(np.isnan(seq)):
        return False
    if label.startswith("fall") and np.std(seq) < 0.02:
        return False
    return True


# =====================================================================
# QUALITY METRICS
# =====================================================================

def compute_quality_scores(seq, gap):
    physics_plaus = float(np.tanh(np.std(seq) / 3.0))
    personalization_strength = float(np.tanh(np.mean(np.abs(seq)) / 5.0))
    domain_alignment = float(gap["mmd"])
    return {
        "physics_plausibility": round(physics_plaus, 4),
        "personalization_strength": round(personalization_strength, 4),
        "domain_alignment": round(domain_alignment, 4)
    }


# =====================================================================
# BOOT
# =====================================================================

print("\n============================================================")
print("  Driftroot Synth Engine — Version 1.0")
print("============================================================")
print(f"Saving simulations → {output_dir}\n")

csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {dataset_path}")

print(f"Found {len(csv_files)} real UP-Fall trials\n")


federator = FederatedSynthesizer()
adapter    = DomainAdapter()

clip_counter  = 0
falls_count   = 0
adls_count    = 0
quality_scores = []


# =====================================================================
# PROCESS EACH CSV TRIAL
# =====================================================================

for csv_path in csv_files:

    filename = os.path.basename(csv_path)
    print(f"\nProcessing → {filename}")

    # -------------------------
    # Error Handling
    # -------------------------
    try:
        seq = from_tof(csv_path)
    except Exception as e:
        print(f"  ❌ Failed to load {filename}: {e}")
        continue

    T = len(seq)
    if T < 20:
        print("  Too few frames → skipping.")
        continue

    print(f"  Loaded {T} frames (10D latent)")


    # --------------------------------------------------------------
    # Progress bar
    # --------------------------------------------------------------
    for i in tqdm(range(SAMPLES_PER_TRIAL), desc="   Generating", ncols=80):

        seq_copy = seq.copy()

        # ---------------------
        # Fall or ADL
        # ---------------------
        if random.random() < 0.5:
            seq_out, label = inject_fall(seq_copy)
            falls_count += 1
        else:
            seq_out, label = seq_copy, "adl"
            adls_count += 1

        # ---------------------
        # Personalization
        # ---------------------
        seq_out = federator.personalize(
            seq_out,
            resident={"age": random.randint(72, 94),
                      "mobility": random.choice(["walker", "cane", "independent"])},
            room={"type": random.choice(["single", "double", "memory_care"]),
                  "layout_noise": random.uniform(0.005, 0.03)},
            sensor={"noise": random.uniform(0.005, 0.02),
                    "drift": random.uniform(-0.01, 0.01)}
        )

        # ---------------------
        # Domain shift
        # ---------------------
        gap = adapter.real_to_synthetic_gap(seq, seq_out)

        # ---------------------
        # Validation
        # ---------------------
        if not validate_generation(seq_out, label):
            continue

        # ---------------------
        # Quality scoring
        # ---------------------
        quality = compute_quality_scores(seq_out, gap)
        quality_scores.append(quality["physics_plausibility"])

        # ---------------------
        # Save
        # ---------------------
        seq_safe = np.clip(seq_out, -1e3, 1e3)

        npy_path  = f"{output_dir}/sim_{clip_counter:05d}.npy"
        json_path = f"{output_dir}/sim_{clip_counter:05d}.json"

        np.save(npy_path, seq_safe.astype(np.float32))

        with open(json_path, "w") as f:
            json.dump({
                "label": label,
                "frames": T,
                "source": "driftroot-synth-v1.0",
                "original_clip": filename,
                "domain_shift": gap,
                "simulation_quality": quality,
                "personalization": {
                    "resident": True,
                    "room": True,
                    "sensor": True
                }
            }, f, indent=2)

        clip_counter += 1


# =====================================================================
# SUMMARY
# =====================================================================

avg_quality = float(np.mean(quality_scores)) if quality_scores else 0.0

print("\n============================================================")
print("      Driftroot v1.0 — Generation Complete")
print("============================================================")
print(f"   Total simulations: {clip_counter}")
print(f"   Falls generated:   {falls_count}")
print(f"   ADLs generated:    {adls_count}")
print(f"   Quality Score:     {avg_quality:.3f}")
print("============================================================\n")