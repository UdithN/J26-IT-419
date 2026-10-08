# J26-IT-419 — Senehasa-Link, Component 2

Gloss-free sign language NLP engine: Sri Lankan Sign Language (SSL) video → Sinhala and English sentences.

## Folder layout

```
data/
  sentences.csv      # ID, Domain, gloss sequence, Sinhala and English targets (UTF-8)
  videos/            # {ID:03d}_t{take}.mp4, e.g. 008_t1.mp4
  keypoints/         # extracted MediaPipe keypoints
src/                 # source code
outputs/
  runs/              # training runs (checkpoints, logs)
  figures/           # plots
tests/               # pytest tests
docs/                # notes and documentation
config.yaml          # all settings (paths, seed, ...)
requirements.txt     # Python packages (torch installed separately)
```

## Setup (Windows 11, Python 3.11)

1. Install torch with CUDA separately (this project uses `torch 2.6.0+cu124`; transformers 5.x needs torch >= 2.6 to load NLLB):
   ```powershell
   python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   ```
2. Install the other packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Check the environment:
   ```powershell
   python src/check_env.py
   ```
