import random
import torch
import os
import librosa
import scipy
import torchaudio.transforms as T
import numpy as np
import torch


def manage_checkpoints(checkpoint_dir, max_checkpoints):
    """
    Keep only the latest 'max_checkpoints' files in the directory.
    """
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')],
                         key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))

    if len(checkpoints) > max_checkpoints:
        for chkpt in checkpoints[:-max_checkpoints]:
            os.remove(os.path.join(checkpoint_dir, chkpt))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def audio_to_melspectrogram(example):
    # The Speech Commands dataset in Hugging Face datasets has the audio array and sampling rate in each example
    audio_array, sampling_rate = example["audio"]["array"], example["audio"]["sampling_rate"]

    # Convert numpy array to tensor
    waveform = torch.tensor(audio_array).float()

    # Ensure waveform is in the shape (num_channels, audio_length)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Define MelSpectrogram transformation
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    # Convert to MelSpectrogram
    mel_spectrogram = mel_spectrogram_transform(waveform)

    # Convert to dB scale
    mel_spectrogram_db = T.AmplitudeToDB()(mel_spectrogram)

    example["mel_spectrogram"] = mel_spectrogram_db.numpy()

    return example


def pad_or_truncate(mel_spec, target_size):
    # mel_spec is expected to be of shape (n_mels, time_frames)
    if mel_spec.shape[-1] > target_size:
        # Truncate
        mel_spec = mel_spec[:, :, :target_size]
    elif mel_spec.shape[-1] < target_size:
        # Pad
        padding_size = target_size - mel_spec.shape[-1]
        mel_spec = torch.nn.functional.pad(mel_spec, (0, padding_size), "constant", 0)
    return mel_spec


def episodic_sampling(c_way, k_shot, base_classes, data):
    ways = random.sample(base_classes.tolist(), c_way)
    support_set = []
    query_set = []
    for way in ways:
        indicies = data[data['label'] == way].index.tolist()
        indicies = random.sample(indicies, k_shot + 1)
        support = torch.stack(
            [pad_or_truncate(torch.tensor(data.iloc[i]['mel_spectrogram']), 32) for i in indicies[:-1]])
        query = pad_or_truncate(torch.tensor(data.iloc[indicies[-1]]['mel_spectrogram']), 32)
        support_set.append(support)
        query_set.append(query)

    return torch.stack(support_set), torch.stack(query_set)


def wav_episodic_sampling(c_way, k_shot, base_classes, data, n_query=1):
    ways = random.sample(base_classes.tolist(), c_way)
    support_set = []
    query_set = []
    wav_set = []
    for way in ways:
        indicies = data[data['label'] == way].index.tolist()
        indicies = random.sample(indicies, k_shot + 1)
        support = torch.stack(
            [pad_or_truncate(torch.tensor(data.iloc[i]['mel_spectrogram']), 32) for i in indicies[:-1]])
        wav = torch.stack(
            [pad_or_truncate(torch.tensor(data.iloc[i]['audio']['array']), 16000) for i in indicies[:-1]])
        query = pad_or_truncate(torch.tensor(data.iloc[indicies[-1]]['mel_spectrogram']), 32)
        support_set.append(support)
        query_set.append(query)
        wav_set.append(wav)
    return torch.stack(support_set), torch.stack(query_set), torch.stack(wav_set)
