import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import librosa
import numpy as np


class ESC50Dataset(Dataset):
    def __init__(self, base_path, meta_data):
        self.base_path = base_path
        self.meta_data = meta_data
        self.sample_rate = 16000  # Target sample rate

        # Mel-spectrogram parameters
        self.n_fft = 2048
        self.hop_length = 497  # Adjusted hop length to get 160 frames
        self.n_mels = 128
        self.max_time_len = int((self.sample_rate * 5) / self.hop_length)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        row = self.meta_data.iloc[idx]
        filename = os.path.join(self.base_path, row.filename)
        label = row.target

        # Load audio using librosa (downsample to 16kHz)
        wav, sr = librosa.load(filename, sr=self.sample_rate)  # [80000 samples]

        # No padding or truncating is performed here

        # Compute Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels  # To avoid internal padding
        )

        # Convert to log scale (dB)
        # mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec_db = np.log(1 + 10000 * mel_spec[:, :self.max_time_len]).reshape(1, self.n_mels, self.max_time_len)

        return torch.tensor(mel_spec_db), label
