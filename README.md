# DeepForestSound (DFS)

Repository accompanying: *DeepForestSound: a multi-species automatic detector for passive acoustic monitoring in African tropical forests*. \
Dubus, G., d’Audiffret, T., Auger, C., Cornette, R., Haupert, S., Kasekendi, I., Katumba, R., Magaldi, H., Pernel, L., Rugonge, H., Sueur, J., Tibesigwa, J.J., Krief, S., 2026. \
 https://doi.org/10.48550/ARXIV.2604.08087

This repository contains scripts to **run inference with pre-trained DFS models** 

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

- Python 3.10 or 3.11 (torch 2.1.0 is not available for Python 3.12+)
- torch==2.1.0, torchaudio==2.1.0
- timm==0.4.5
- peft==0.10.0, transformers==4.40.2
- numpy, pandas, soundfile, tqdm, wget, matplotlib

Runs with or without an NVIDIA GPU.

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

Main options:

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--audio_dir` | *(required)* | Folder containing the `.wav` / `.WAV` files to analyse |
| `--output_dir` | `outputs/csv` | Folder where the CSV files are written |
| `--weights_lf` / `--weights_mf` | `model_weights/DFS_LF.pth` / `model_weights/DFS_MF.pth` | Paths to the model weights |
| `--device` | `auto` | `auto` (GPU if available, otherwise CPU), `cuda` or `cpu` |
| `--duration` | `10` | Chunk duration in seconds |
| `--hop` | `10` | Hop between chunks in seconds |
| `--batch_size` | `16` | Number of chunks processed at once (lower it if you run out of memory) |

Running on CPU is supported (about 6–10 s per one-minute file on a standard computer without GPU).

Output:
One CSV file per audio file, saved in `--output_dir`. Each row is one chunk, with its start and end time in seconds (`time_in`, `time_out`) and the score (0–1) of each species. The last chunk of a file is aligned on the end of the recording, so every second of audio is analysed without padding.

## Suggested detection thresholds

The CSV files contain raw scores between 0 and 1. To turn them into detections (presence / absence per chunk), a threshold has to be applied to each species. The following species-specific thresholds are suggested:

| Species | Column name | Threshold |
| ------- | ----------- | --------- |
| *Loxodonta cyclotis* (African forest elephant) | `loxodonta` | 0.30 |
| *Balearica regulorum* | `balearica_regulorum` | 0.82 |
| *Bycanistes subcylindricus* | `bycanistes_subcylindricus` | 0.33 |
| *Chrysococcyx cupreus* | `chrysococcyx_cupreus` | 0.10 |
| *Colobus guereza* | `colobus_guereza` | 0.71 |
| *Corythaeola cristata* | `corythaeola_cristata` | 0.44 |
| *Laniarius mufumbiri* | `laniarius_mufumbiri` | 0.41 |
| *Lophocebus albigena* | `lophocebus_albigena` | 0.80 |
| *Pan troglodytes* | `pan_troglodytes` | 0.99 |
| *Streptopelia semitorquata* | `streptopelia_semitorquata` | 0.20 |
| *Tauraco schuettii* | `tauraco_schuettii` | 0.71 |
| *Turtur tympanistria* | `turtur_tympanistria` | 0.10 |

These values are a starting point. Depending on your site and objectives, you may lower a threshold to miss fewer vocalizations (more false positives) or raise it to get fewer false positives (more missed vocalizations). Checking a sample of detections by listening or looking at spectrograms is recommended.

Example to apply them to the output CSV files:
```python
import glob
import os
import pandas as pd

csv_dir = "outputs/csv"
thresholds = {
    "loxodonta": 0.30, "balearica_regulorum": 0.82, "bycanistes_subcylindricus": 0.33,
    "chrysococcyx_cupreus": 0.10, "colobus_guereza": 0.71, "corythaeola_cristata": 0.44,
    "laniarius_mufumbiri": 0.41, "lophocebus_albigena": 0.80, "pan_troglodytes": 0.99,
    "streptopelia_semitorquata": 0.20, "tauraco_schuettii": 0.71, "turtur_tympanistria": 0.10,
}

detections = []
for path in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
    df = pd.read_csv(path)
    for sp, thr in thresholds.items():
        hits = df.loc[df[sp] >= thr, ["time_in", "time_out", sp]].rename(columns={sp: "score"})
        hits.insert(0, "species", sp)
        hits.insert(0, "file", os.path.basename(path))
        detections.append(hits)

detections = pd.concat(detections, ignore_index=True)
detections.to_csv(os.path.join(csv_dir, "detections.csv"), index=False)
```

## GUI

A standalone graphical user interface (GUI) is available for users who prefer to run DFS without using the command line.

Two versions are provided:

* **CPU version** — for computers without a compatible NVIDIA GPU
* **GPU version** — recommended for computers with a compatible NVIDIA GPU, for faster inference

| Version     | Download                               |
| ----------- | -------------------------------------- |
| **CPU** | [Download DFS GUI – CPU](https://drive.google.com/file/d/1YsK9eOL4VLuXbKJ4vF4rFLKxdpbtiPPF/view?usp=sharing) |
| **GPU**  | [Download DFS GUI – GPU](https://drive.google.com/drive/folders/1mniVrXx8_0mgGMqVJF0fe3lV4nG9khZb?usp=sharing) |

> **Note:** The GUI is provided as a standalone executable and does not require a Python installation.

## Test data and example outputs

To facilitate testing and allow users to verify the expected outputs, a set of **7 one-minute audio files** is provided:
 **[Download the 7 test audio files](https://drive.google.com/drive/folders/1Fz-aaLVCL39o6YRTNeYXo_1iixI14Mgb?usp=sharing)**

These files can be used directly with the DFS GUI to test the inference pipeline and compare the generated results with the provided reference outputs.

Example outputs obtained with these test files are also provided: **[Download the example outputs](https://drive.google.com/drive/folders/1BYlkCeX7myPssOeSgu858mCSvCuuz6RQ?usp=sharing)**

The example outputs include:

* **CSV files** containing the detection results and predictions for each audio file
* **Windowed audio files** corresponding to detected positive chunks
* **Spectrograms** of the positive chunks

These files provide reference results for checking that the GUI is correctly installed and that the inference pipeline produces the expected outputs.

## References

[1]: Gong, Y., Chung, Y.-A., Glass, J., 2021. AST: Audio Spectrogram Transformer, in: Interspeech 2021. Presented at the Interspeech 2021, p. 575. https://doi.org/10.21437/Interspeech.2021-698

[2]: Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., 2022. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 1, 3. https://doi.org/10.48550/arXiv.2106.09685

[3]: https://xeno-canto.org/, visited on Jan. 2026.

[4]: Zwerts, J.A., Treep, J., Kaandorp, C.S., Meewis, F., Koot, A.C., Kaya, H., 2021. Introducing a Central African Primate Vocalisation Dataset for Automated Species Classification, in: Interspeech 2021. Presented at the Interspeech 2021, ISCA, pp. 466–470. https://doi.org/10.21437/Interspeech.2021-154

[5]: https://www.elephantlisteningproject.org/congo-soundscapes-public-database/, visited on Jan. 2026.