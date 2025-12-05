# Synthetic Fall Engine v1.0

![Version](https://img.shields.io/badge/version-v1.0-blue)
![License](https://img.shields.io/badge/license-Apache_2.0-green)
![Python](https://img.shields.io/badge/python-3.14%2B-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)

Synthetic Fall Engine v1.0 is a physics informed synthetic fall data generator designed for Time-of-Flight (ToF) sensors. It converts real UP-Fall motion recordings into thousands of personalized synthetic fall and ADL sequences using physics based fall injection, personalized resident/environment/sensor modeling, and domain-shift scoring. Outputs include latent motion arrays and metadata-rich JSON files. The engine is fully reproducible using a fixed global seed.

Example metadata:
```json
{
  "label": "fall_forward",
  "frames": 487,
  "source": "synth-v1.0",
  "original_clip": "Subject1Activity08Trial2.csv",
  "domain_shift": { "mmd": 14.72, "coral": 16284.11 },
  "simulation_quality": {
    "physics_plausibility": 0.98,
    "personalization_strength": 0.94,
    "domain_alignment": 14.72
  },
  "personalization": { "resident": true, "room": true, "sensor": true }
}
```

Install:
```bash
pip install numpy scipy scikit-learn tqdm
```

Run:
```bash
python pipeline.py
```

Outputs are saved in:
```
data/simulations/
```

## **Background**

This project was my early exploration into synthetic data and simulation-driven motion modeling before pivoting into edge AI systems.

