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