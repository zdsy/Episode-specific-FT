"""Code to pick a random augmentation from a list and apply it to audio"""
import torch
import random
import numpy as np
from torchaudio.functional import equalizer_biquad
import torchaudio
import pandas as pd
import soundfile as sf
import os
import librosa


class RandAugment:
  def __init__(self,
               aug_methods,
               aug_scale,
               rir_df_path,
               sample_rate):
    """Apply a random augmentation from a list of augmentation on raw waveform

    Parameters
    ----------
    aug_methods : list
        List of the augmentation method. [Time shift, Pitch shift, Filter,
        Reverberation, Compression, Noise, Overdrive]
    aug_scale : int
        Intensity of the augmentation, range from 1 to 5.
    rir_df_path : str
        Path to a folder containing room impulse responses.
    sample_rate : int
        Sample rate at which we operate.
    """
    self.sample_rate = sample_rate

    # Random augmentation methods
    augment_list = []
    if "Time shift" in aug_methods:
      augment_list.append(self.time_shift)
      print("Using time shift")
    if "Pitch shift" in aug_methods:
      augment_list.append(self.pitch_shift)
      print("Using pitch shift")
    if "Filter" in aug_methods:
      augment_list.append(self.filter_aug)
      print("Using filter")
    if "Reverberation" in aug_methods:
      self.rir_df = pd.read_csv(os.path.expandvars(
          rir_df_path), header=0, sep="\t")
      augment_list.append(self.reverberation)
      self.rir = self.load_rir(self.rir_df)
      print("Using reverberation")
    if "Compression" in aug_methods:
      augment_list.append(self.compression)
      print("Using compression")
    if "Noise" in aug_methods:
      augment_list.append(self.noise)
      print("Using noise")
    if "Overdrive" in aug_methods:
      augment_list.append(self.overdrive)
      print("Using overdrive")
    if not augment_list:
      self.augment_list = None
    else:
      self.augment_list = augment_list

    self.augment_scale = aug_scale
    print("RandAugment scale:", self.augment_scale)

  def selector(self, augment_list):
    return np.random.choice(augment_list)

  def random_augment(self, feats: torch.FloatTensor):

    if self.augment_list is not None:
      aug_feats, distortions = self._augment(feats.clone())
      return aug_feats, distortions

    else:
      return feats, ()

  def _augment(self, data):

    # Random augmentation
    randaug_func = self.selector(self.augment_list)
    aug_data, distortion_randaug = randaug_func(
        utt=data)

    return aug_data, distortion_randaug

  def time_shift(self, utt: torch.FloatTensor):
    """Time shift utterance in time domain.
    Input:
        Utterances              -   [num_samples (Sr * Ts)]
    Output:
        Augmented utterance     -   [num_samples (Sr * Ts)]
        Distortion              -   ("Time shift", None)
    """
    with torch.no_grad():
      scale = random.randint(1, self.augment_scale)
      shift_scale = scale / 1000
      # Shifting data in time-domain
      utt_split = int(utt.shape[1] * shift_scale)
      utt_aug = torch.cat(
          [utt[:, -utt_split:], utt[:, :-utt_split]], dim=-1)

    return utt_aug, ("Time shift", shift_scale)

  def pitch_shift(self, utt: torch.FloatTensor):
    """Pitch shifting by vocoder method inside sox.
    Input:
        Utterances              -   [num_samples (Sr * Ts)]
    Output:
        Augmented utterance     -   [num_samples (Sr * Ts)]
        Distortion              -   ("Pitch shift", n_bin shifted)
    """
    scale = random.randint(int(-self.augment_scale/2),
                           int(self.augment_scale/2))
    if scale == 0:
      scale = self.rand_sign()

    # transform = torchaudio.transforms.PitchShift(
    #     sample_rate=self.sample_rate,
    #     n_steps=scale)

    # transform.eval()
    #
    # with torch.no_grad():
    #
    #   utt = transform(utt)

    utt = librosa.effects.pitch_shift(utt.numpy(), sr=self.sample_rate, n_steps=scale)

    return torch.tensor(utt), ("Pitch shift", scale)

  def filter_aug(self, utt: torch.FloatTensor):
    """Filter augmentation.
    Input:
        Utterances              -   [num_samples (Sr * Ts)]
    Output:
        Augmented utterance     -   [num_samples (Sr * Ts)]
        Distortion              -   ("Filter", None)
    """
    scale = self.augment_scale

    n_bands = torch.randint(low=2, high=5, size=(1,))
    center_freqs = torch.randint(low=30, high=3000, size=(n_bands,))
    # To have gains between -10 and 10 db
    gains = (torch.rand(n_bands) * 20 - 10) * scale/5

    for i in range(n_bands):

      utt = equalizer_biquad(waveform=utt,
                             sample_rate=self.sample_rate,
                             center_freq=center_freqs[i],
                             gain=gains[i],
                             Q=0.7)

    return utt, ("Filter", (n_bands, center_freqs, gains))

  def reverberation(self, utt: torch.FloatTensor):
    """Reverberation augmentation.
    Input:
        Utterances              -   [num_samples (Sr * Ts)]
    Output:
        Augmented utterance     -   [num_samples (Sr * Ts)]
        Distortion              -   ("Reverberation", None)
    """

    with torch.no_grad():
      scale = random.randint(1, self.augment_scale) / \
          (3.33*self.augment_scale)

      # select random index
      index = np.random.randint(len(self.rir))

      # load rir
      rir = self.rir[index]

      utt_reverb = torchaudio.functional.convolve(
          utt, rir[None, :], mode="full")

      size_rir = rir.shape[0]

      utt_reverb_tronc = utt_reverb[:, :-size_rir+1]

      utt = utt_reverb_tronc*scale + utt*(1-scale)
    return utt, ("Reverberation", scale)

  def compression(self, utt: torch.FloatTensor):
    with torch.no_grad():
      scale = self.augment_scale
      enhancement_amount = np.random.randint(low=0, high=scale*2)
      utt = torchaudio.functional.contrast(utt, enhancement_amount)
    return utt, ("Compression", enhancement_amount)

  def noise(self, utt: torch.FloatTensor):
    # transform hyperparameters

    scale = self.augment_scale
    sr = self.sample_rate
    snr_min = 12
    snr_max = 100 * 1/scale

    with torch.no_grad():

      # generate noise
      noise = torch.normal(torch.zeros(
          sr), torch.ones(sr)).to(utt.device)

      # generate psd profile
      f_decay = np.random.choice([0, 1, 2, -1, -2])
      spec = torch.fft.rfft(noise)
      mask = torch.pow(torch.linspace(1, (sr / 2) ** 0.5,
                       spec.shape[0]).to(utt.device), -f_decay)

      # apply psd
      spec *= mask
      noise = torch.fft.irfft(spec).reshape(1, -1).squeeze()
      noise /= torch.sqrt(torch.mean(torch.square(noise)))

      # get correct shape
      noise = torch.cat(
          [noise] * int(np.ceil(utt.shape[0]/noise.shape[0])), axis=0)[:utt.shape[0]]

      # scale noise
      snr = np.random.uniform(
          snr_min, snr_max)
      gain = torch.sqrt(torch.square(utt).sum() /
                        (10 ** (snr / 10) * torch.square(noise).sum()))
      noise *= gain

      # add noise
      utt += noise

    return utt, ("Noise", snr)

  def overdrive(self, utt: torch.FloatTensor):
    # transform hyperparameters
    min_gain = 0.3
    max_gain = 1.0
    min_colour = 0.3
    max_colour = 1.0

    with torch.no_grad():
      scale = self.augment_scale

      gain = np.random.uniform(
          low=min_gain, high=min_gain + (max_gain - min_gain) / 5 * scale)
      colour = np.random.uniform(
          low=min_colour, high=min_colour + (max_colour - min_colour)/5*scale)
      utt = torchaudio.functional.overdrive(
          utt, gain=gain, colour=colour)

    return utt, ("Overdrive", (gain, colour))

  def load_rir(self, rir_df):
    rir_list = []

    for rir_path in rir_df["filepath"]:
      rir, sr = sf.read(rir_path)

      # reduce channels
      if rir.ndim > 1:
        rir = np.mean(rir, axis=1)

      # Normalizing rir
      rir = torch.from_numpy(rir).float()
      rir = torch.nn.functional.normalize(rir, dim=-1)

      # resample
      if sr != self.sample_rate:
        # Parameters to match Kaiser Best of librosa
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=self.sample_rate,
            resampling_method="sinc_interp_kaiser",
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            beta=14.769656459379492)
        rir = resampler(rir)

      rir_list.append(rir)

    return rir_list

  @staticmethod
  def rand_sign():
    return 1 if torch.randn(1) > 0.5 else -1


if __name__ == "__main__":

  import time

  aug_methods = ["Noise"]
  # aug_methods = ["Reverberation", "Overdrive"]
  rir_df_path = "/home/auquelennec/Téléchargements/but/local_rir.tsv"
  aug_scale = 2

  augmentor = RandAugment(aug_methods=aug_methods,
                          aug_scale=aug_scale,
                          rir_df_path=rir_df_path,
                          sample_rate=16000)

  audio_fp = "/home/auquelennec/Bureau/0/0DYH0sqDXB8_30.000_40.000.wav"

  wav, curr_sample_rate = sf.read(audio_fp, dtype="float32")

  feats = torch.from_numpy(wav).float()

  feats = feats.unsqueeze(dim=0)
  mean_time = []

  for i in range(10):
    start = time.time()
    audio, distortions = augmentor.random_augment(feats=feats)
    stop = time.time()
    mean_time.append(stop-start)

  print("Time for {} : {}".format(aug_methods[0], np.mean(mean_time)))

  # sf.write("/home/auquelennec/Bureau/aug_audio.wav", audio, 16000)