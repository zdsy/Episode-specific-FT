import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class MedleySolosDBDataset(Dataset):
    def __init__(self, base_path, metadata_file, sample_rate=16000):
        self.base_path = base_path
        self.meta_data = pd.read_csv(metadata_file)

        # Filter by subset (training, validation, test)
#         self.meta_data = self.meta_data[self.meta_data['subset'] == target_subset].reset_index(drop=True)

        self.sample_rate = sample_rate  # Target sample rate
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.max_time_len = int((self.sample_rate * 3) / self.hop_length)  # 3 seconds

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        row = self.meta_data.iloc[idx]
        subset = row['subset']
        instrument_id = row['instrument_id']
        uuid = row['uuid4']

        # Build filename
        filename = f"Medley-solos-DB_{subset}-{instrument_id}_{uuid}.wav"
        filepath = os.path.join(self.base_path, 'audio', filename)

        # Load audio (downsample to 16kHz)
        wav, sr = librosa.load(filepath, sr=self.sample_rate)

        # Compute Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # Convert to log scale and clip
        mel_spec_db = np.log(1 + 10000 * mel_spec[:, :self.max_time_len]).reshape(1, self.n_mels, self.max_time_len)

        label = instrument_id

        return torch.tensor(mel_spec_db, dtype=torch.float32), label, torch.tensor(wav)