import os
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import learn2learn as l2l
from model import MatchingNetwork
from tqdm import tqdm
import librosa
import torch
import numpy as np
from wav_data_loader import ESC50Dataset
from utils import wav_episodic_sampling, one_hot, mean_confidence_interval
from aug import RandAugment

# Load the metadata file
meta_data = pd.read_csv('../ESC-50/meta/esc50.csv')

np.random.seed(66)
classes = np.arange(50)
np.random.shuffle(classes)
train_classes = classes[:35]
val_classes = classes[35:40]
test_classes = classes[40:]
train_meta = meta_data[meta_data.target.isin(train_classes)]
val_meta = meta_data[meta_data.target.isin(val_classes)]
test_meta = meta_data[meta_data.target.isin(test_classes)]

train_dataset = ESC50Dataset('../ESC-50/audio', train_meta)
val_dataset = ESC50Dataset('../ESC-50/audio', val_meta)
test_dataset = ESC50Dataset('../ESC-50/audio', test_meta)

train_dataset.meta_data = train_dataset.meta_data.reset_index(drop=True)
val_dataset.meta_data = val_dataset.meta_data.reset_index(drop=True)
test_dataset.meta_data = test_dataset.meta_data.reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MatchingNetwork.MatchingNetwork(batch_size=8,
                                        keep_prob=torch.torch.FloatTensor(1), num_channels=1,
                                        fce=False,
                                        num_classes_per_set=5,
                                        num_samples_per_class=5,
                                        nClasses=0, image_size=28).to(device)
mcmn = l2l.algorithms.GBML(
    module=model,
    transform=l2l.optim.MetaCurvatureTransform,
    lr=0.2,
    adapt_transform=False,
    first_order=True,  # has both 1st and 2nd order versions
).to(device)

augmentor = RandAugment(aug_methods=['Noise', 'Filter', 'Pitch shift'],
                        aug_scale=4,
                        rir_df_path='',
                        sample_rate=16000)


def aug_wav_2_mel(wav, augmentor):
    aug, _ = augmentor.pitch_shift(wav)
    aug_mel_spec = librosa.feature.melspectrogram(
        y=aug.numpy(),
        sr=16000,
        n_fft=2048,
        hop_length=497,
        n_mels=128  # To avoid internal padding
    )
    return torch.tensor(
        np.log(1 + 10000 * aug_mel_spec[:, :int((16000 * 5) / 497)]).reshape(1, 128, int((16000 * 5) / 497)))


checkpoint_path = ''  # Update this path
checkpoint = torch.load(checkpoint_path)
mcmn.load_state_dict(checkpoint['model_state_dict'])

val_eps = 5000
batch_size = 1
steps = 8

# pseudo-label
slabel = one_hot(torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]))
qlabel = torch.tensor([0, 1, 2, 3, 4])
slabel_batch = []
qlabel_batch = []
for b in range(batch_size):
    slabel_batch.append(slabel)
    qlabel_batch.append(qlabel)
slabel_batch = torch.stack(slabel_batch)
qlabel_batch = torch.stack(qlabel_batch)

PRE_VAL_ACC = []
POST_VAL_ACC = []
TR_VAL_ACC = []
for val_episode in tqdm(range(val_eps), leave=True):
    # Clone MC model
    val_model = mcmn.clone()

    # Prepare batch data
    sbatch_val = []
    qbatch_val = []
    wav_batch_val = []
    for b in range(batch_size):
        s, q, wav = wav_episodic_sampling(5, 5, test_classes, test_dataset, n_query=10)
        sbatch_val.append(s)
        qbatch_val.append(q)
        wav_batch_val.append(wav)
    sbatch_val = torch.stack(sbatch_val)
    qbatch_val = torch.stack(qbatch_val)
    wav_batch_val = torch.stack(wav_batch_val)

    # Calculate Pre Acc
    with torch.no_grad():
        for i in range(10):
            pre_val_acc, _ = val_model.module(sbatch_val.view(batch_size, 25, 1, 128, 160).cuda(), slabel_batch.cuda(),
                                              qbatch_val[:, :, i].view(batch_size, 5, 1, 128, 160).cuda(),
                                              qlabel_batch.cuda())
            PRE_VAL_ACC.append(pre_val_acc.item())

    for step in range(steps):
        # ADFT
        sbatch = sbatch_val.view(5, 5, 1, 128, 160)
        wav_batch = wav_batch_val.squeeze(0)

        for j in range(sbatch.size(1)):
            s_f = torch.stack([
                torch.cat((
                    torch.cat((sbatch[i, :j], sbatch[i, j + 1:])),  # Removes the j-th column
                    aug_wav_2_mel(torch.cat((wav_batch[i, :j], wav_batch[i, j + 1:]))[j - 1 if j > 0 else -1],
                                  augmentor).unsqueeze(0)
                ))
                for i in range(sbatch.size(0))])
            q_f = torch.stack([sbatch[i, j] for i in range(sbatch.size(0))])

            tr_val_acc, val_loss = val_model.module(s_f.view(batch_size, 25, 1, 128, 160).cuda(),
                                                    slabel_batch.cuda(),
                                                    q_f.view(batch_size, 5, 1, 128, 160).cuda(), qlabel_batch.cuda())
            val_model.adapt(val_loss)
            TR_VAL_ACC.append(tr_val_acc.item())

    # Calculate Post Acc
    for i in range(10):
        post_val_acc, post_val_loss = val_model.module(sbatch_val.view(batch_size, 25, 1, 128, 160).cuda(),
                                                       slabel_batch.cuda(),
                                                       qbatch_val[:, :, i].view(batch_size, 5, 1, 128, 160).cuda(),
                                                       qlabel_batch.cuda())
        POST_VAL_ACC.append(post_val_acc.item())

    print(f'pre_val_acc: {np.mean(PRE_VAL_ACC):.4f}, post_val_acc: {np.mean(POST_VAL_ACC):.4f}')

print(mean_confidence_interval(PRE_VAL_ACC))
print(mean_confidence_interval(POST_VAL_ACC))
