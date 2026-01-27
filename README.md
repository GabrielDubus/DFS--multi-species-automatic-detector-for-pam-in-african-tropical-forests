# DeepForestSound (DFS)

Repository accompanying: *DeepForestSound: a multi-species automatic detector for passive acoustic monitoring in African tropical forests* (IEEE).

This repository contains scripts to **run inference on pre-trained DFS models** 

## Background

**DeepForestSound (DFS)** is a multi-species automatic detection model designed for passive acoustic monitoring (PAM) in African tropical forests. 

DFS is based on:

- **Audio Spectrogram Transformer (AST)** architecture [Gong et al., Interspeech 2021]
- **Low-Rank Adaptation (LoRA)** for efficient fine-tuning on limited annotated datasets [Hu et al., ICLR 2022]
- Dual-frequency models:  
  - LF (Low-Frequency) for elephant rumbles  
  - MF (Mid-Frequency) for other birds and primates

DFS was trained on a combination of Sebitoli 2023 recordings (16 Song Meter Mini recorders capturing morning and afternoon vocal activity), publicly available datasets including Xeno-Canto (bird vocalizations), the Central African Primate Vocalization Dataset, and Congo Soundscapes (elephant rumbles), as well as additional species-specific datasets for chimpanzees and elephants collected in Sebitoli. The model was evaluated on independent Sebitoli 2025 recordings from new locations within the same area, covering a 6-month period.

## Repository Structure
```
DFS-GitHub/
│
├── README.md
├── requirements.txt
├── run_inference.py
├── src/
│   ├── __init__.py
│   ├── ast_models.py
│   ├── LoRA_inject.py
│   └── utils.py
├── model_weights/
└── outputs/
```

## Installation

Tested with:

- Python 3.10+
- torch >= 2.0
- timm==0.4.5
- numpy, pandas, librosa, scipy, audioread, soundfile, tqdm, peft, wget, matplotlib


Install with:
```bash
pip install -r requirements.txt
```
## Usage
1. Run inference
```bash
python run_inference.py \
  --audio_dir datasets/EVAL/audio \
  --weights_lf model_weights/eleph_mlphead.pth \
  --weights_mf model_weights/mf11_mlphead.pth \
  --output_dir outputs/EVAL
```

Output:
CSV files for each audio file, saved in the specified `--output_dir` (default: `outputs/`). Each CSV contains per-chunk predictions for all species.


## Notes
LoRA adapters are used for fine-tuning; the rest of AST is frozen.
Supports multi-label detection (each audio chunk may contain multiple species).
Designed for reproducible evaluation in African tropical forest datasets.

