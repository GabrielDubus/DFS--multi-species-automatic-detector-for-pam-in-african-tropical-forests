# DeepForestSound (DFS)

Repository accompanying: *DeepForestSound: a multi-species automatic detector for passive acoustic monitoring in African tropical forests*.

This repository contains scripts to **run inference on pre-trained DFS models** 

## Background

**DeepForestSound (DFS)** is a multi-species automatic detection model designed for passive acoustic monitoring (PAM) in African tropical forests. 

DFS is based on:

- **Audio Spectrogram Transformer (AST)** architecture[1]
- **Low-Rank Adaptation (LoRA)** for efficient fine-tuning on limited annotated datasets[2]
- Dual-frequency models:  
  - LF (Low-Frequency) for elephant rumbles  
  - MF (Mid-Frequency) for other birds and primates

DFS was trained on a combination of passive acoustic data recorded in Sebitoli (North of Kibale National Park, Uganda) in 2023, publicly available datasets including Xeno-Canto[3], the Central African Primate Vocalization Dataset[4], and extracted data from Congo Soundscapes – Public Database[5], as well as additional species-specific datasets for chimpanzees and elephants collected in Sebitoli. The model was evaluated on independent Sebitoli 2025 recordings from new locations within the same forest.



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
  --audio_dir path/to/your/audiofiles \
  --weights_lf model_weights/DFS_LF.pth \
  --weights_mf model_weights/DFS_MF.pth \
  --output_dir outputs/
```

Output:
CSV files for each audio file, saved in the specified `--output_dir` (default: `outputs/`). Each CSV contains per-chunk predictions for all species.



## References

[1]: Gong, Y., Chung, Y.-A., Glass, J., 2021. AST: Audio Spectrogram Transformer, in: Interspeech 2021. Presented at the Interspeech 2021, p. 575. https://doi.org/10.21437/Interspeech.2021-698

[2]: Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., 2022. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 1, 3. https://doi.org/10.48550/arXiv.2106.09685

[3]: https://xeno-canto.org/, visited on Jan. 2026.

[4]: Zwerts, J.A., Treep, J., Kaandorp, C.S., Meewis, F., Koot, A.C., Kaya, H., 2021. Introducing a Central African Primate Vocalisation Dataset for Automated Species Classification, in: Interspeech 2021. Presented at the Interspeech 2021, ISCA, pp. 466–470. https://doi.org/10.21437/Interspeech.2021-154

[5]: https://www.elephantlisteningproject.org/congo-soundscapes-public-database/, visited on Jan. 2026.