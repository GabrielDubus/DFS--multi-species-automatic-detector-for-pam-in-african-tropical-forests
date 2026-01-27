#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for DeepForestSound (DFS) inference.
Includes audio loading, feature extraction, and optional plotting.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


def make_features_lf(
    waveform, sr, mel_bins=128, target_length=256, 
    apply_masking=False, time_mask_param=15, freq_mask_param=20,
    apply_propagation_sim=False, prop_attenuation_factor=0.5
):
    """
    Generate low-frequency (LF) log-mel features for elephant rumbles.
    """
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)
    waveform = waveform.astype(np.float32)
    waveform = (waveform - np.mean(waveform)) / np.std(waveform)
    waveform = torch.from_numpy(waveform)[None, :]

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_length=750, frame_shift=35, low_freq=3, high_freq=250
    )

    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
    elif p < 0:
        fbank = fbank[:target_length, :]

    if apply_propagation_sim:
        freqs = torch.linspace(0, 1, mel_bins)
        attenuation_curve = torch.exp(-prop_attenuation_factor * freqs)
        fbank = fbank * attenuation_curve.unsqueeze(0)

    # Normalize
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)

    if apply_masking:
        augment = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        )
        fbank = augment(fbank)

    return fbank


def make_features_mf(
    waveform, sr, mel_bins=128, target_length=512, 
    apply_masking=False, time_mask_param=15, freq_mask_param=20,
    apply_propagation_sim=False, prop_attenuation_factor=10
):
    """
    Generate mid-frequency (MF) log-mel features for birds and primates.
    """
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)
    waveform = waveform.astype(np.float32)
    waveform = (waveform - np.mean(waveform)) / np.std(waveform)
    waveform = torch.from_numpy(waveform)[None, :]

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_length=100, frame_shift=19, low_freq=50, high_freq=4000
    )

    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
    elif p < 0:
        fbank = fbank[:target_length, :]

    if apply_propagation_sim:
        attenuation_curve = torch.linspace(0, 1, mel_bins) * prop_attenuation_factor
        fbank = fbank - attenuation_curve.unsqueeze(0)

    # Normalize
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)

    if apply_masking:
        augment = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        )
        fbank = augment(fbank)

    return fbank


def load_audio(wav_path, sr, offset, duration):
    """
    Load a segment of audio as a numpy array.
    Args:
        wav_path: path to the audio file
        sr: target sampling rate
        offset: start time in seconds
        duration: segment duration in seconds
    Returns:
        waveform: 1D numpy array of shape (samples,)
        sr: sampling rate
    """
    try:
        info = sf.info(wav_path)
        orig_sr = info.samplerate
        total_duration = info.duration

        if offset + duration > total_duration:
            offset = max(0, total_duration - duration)

        frame_offset = int(offset * orig_sr)
        num_frames = int(duration * orig_sr)

        waveform, file_sr = torchaudio.load(wav_path, frame_offset=frame_offset, num_frames=num_frames)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if file_sr != sr:
            resampler = T.Resample(orig_freq=file_sr, new_freq=sr)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0).numpy()

        target_len = int(sr * duration)
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        elif len(waveform) > target_len:
            waveform = waveform[:target_len]

        return waveform, sr

    except Exception as e:
        print(f"Failed to load {wav_path} at offset {offset}: {e}")
        return None


def plot_features(feats, title="Feature", savepath="feat.png"):
    """
    Plot log-mel features for visualization.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    spec = feats.numpy().T
    im = ax.imshow(spec, origin='lower', aspect='auto', cmap='magma')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(savepath)
    plt.close()
