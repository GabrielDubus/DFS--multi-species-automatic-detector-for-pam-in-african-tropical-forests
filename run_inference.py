# -*- coding: utf-8 -*-
"""
Command-line inference script for AST + LoRA models.
Runs sliding-window inference on an evaluation audio set and exports CSV files.
"""

import os
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import audioread
import torch
from tqdm import tqdm


from src import ast_models
from src import utils
from src.LoRA_inject import inject_lora


# ============================================================
# Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AST + LoRA inference on an audio evaluation set"
    )

    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing .wav evaluation files")

    parser.add_argument("--output_dir", type=str, default="outputs/csv",
                        help="Directory to store CSV outputs")
    
    parser.add_argument("--weights_lf", type=str, default='model_weights/DFS_LF.pth',
                        help="Path to low-frequency (elephant) model weights")
    
    parser.add_argument("--weights_mf", type=str, default='model_weights/DFS_MF.pth',
                        help="Path to mid-frequency model weights")


    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Computation device")

    parser.add_argument("--duration", type=float, default=10.0,
                        help="Chunk duration in seconds")

    parser.add_argument("--hop", type=float, default=10.0,
                        help="Hop size between chunks in seconds")

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    DEVICE = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    MEL_BINS = 128

    SR_LF = 1000
    TL_LF = 256

    SR_MF = 8000
    TL_MF = 512

    FSTRIDE = 10
    TSTRIDE = 10

    SPECIES = [
        'loxodonta',
        'balearica_regulorum',
        'bycanistes_subcylindricus',
        'chrysococcyx_cupreus',
        'colobus_guereza',
        'corythaeola_cristata',
        'laniarius_mufumbiri',
        'lophocebus_albigena',
        'pan_troglodytes',
        'streptopelia_semitorquata',
        'tauraco_schuettii',
        'turtur_tympanistria'
    ]

    # --------------------------------------------------------
    # Load models + LoRA
    # --------------------------------------------------------

    model_lf = ast_models.ASTModel(
        label_dim=1,
        fstride=FSTRIDE,
        tstride=TSTRIDE,
        input_fdim=MEL_BINS,
        input_tdim=TL_LF,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384"
    )

    model_mf = ast_models.ASTModel(
        label_dim=len(SPECIES) - 1,
        fstride=FSTRIDE,
        tstride=TSTRIDE,
        input_fdim=MEL_BINS,
        input_tdim=TL_MF,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384"
    )

    model_lf = inject_lora(model_lf)
    model_mf = inject_lora(model_mf)

    state_dict_lf = torch.load(args.weights_lf, map_location=DEVICE)
    model_lf.load_state_dict(state_dict_lf)
    state_dict_mf = torch.load(args.weights_mf, map_location=DEVICE)
    model_mf.load_state_dict(state_dict_mf)

    model_lf.eval().to(DEVICE)
    model_mf.eval().to(DEVICE)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    audio_files = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))

    for wav_path in tqdm(audio_files, desc="Running inference"):

        out_csv = Path(args.output_dir) / (Path(wav_path).stem + ".csv")
        if out_csv.exists():
            continue

        try:
            total_dur = audioread.audio_open(wav_path).duration
        except Exception:
            continue

        n_chunks = max(
            1,
            int((total_dur - args.duration) / args.hop) + 2
        )

        feats_lf, feats_mf, times = [], [], []

        for i in range(n_chunks):
            offset = i * args.hop

            x_lf, _ = utils.load_audio(
                wav_path, SR_LF, offset, args.duration
            )
            x_mf, _ = utils.load_audio(
                wav_path, SR_MF, offset, args.duration
            )

            if len(x_lf) < SR_LF * args.duration:
                x_lf = np.pad(
                    x_lf, (0, int(SR_LF * args.duration) - len(x_lf))
                )
                x_mf = np.pad(
                    x_mf, (0, int(SR_MF * args.duration) - len(x_mf))
                )

            feats_lf.append(
                utils.make_features_lf(
                    x_lf, SR_LF, MEL_BINS, TL_LF
                )
            )
            feats_mf.append(
                utils.make_features_mf(
                    x_mf, SR_MF, MEL_BINS, TL_MF
                )
            )

            times.append(offset)

        with torch.no_grad():
            X_lf = torch.tensor(np.stack(feats_lf)).float().to(DEVICE)
            X_mf = torch.tensor(np.stack(feats_mf)).float().to(DEVICE)

            y_lf = torch.sigmoid(model_lf(X_lf))
            y_mf = torch.sigmoid(model_mf(X_mf))

            scores = torch.cat([y_lf, y_mf], dim=1).cpu().numpy()

        df = pd.DataFrame({
            "time_in": times,
            "time_out": args.duration
        })

        for i, sp in enumerate(SPECIES):
            df[sp] = scores[:, i]

        df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
