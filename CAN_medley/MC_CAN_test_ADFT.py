import os
import pandas as pd
import torch
from utils import wav_episodic_sampling, one_hot, mean_confidence_interval
from wav_data_loader import MedleySolosDBDataset
import librosa
import numpy as np
import random
import os
from tqdm import tqdm
from model.net import Model
import torch.nn as nn
from model.losses import CrossEntropyLoss
import learn2learn as l2l
from aug import RandAugment

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

base_path = '../Medley-solo'
metadata_file = os.path.join(base_path, 'metadata.csv')

dataset = MedleySolosDBDataset(base_path, metadata_file)

train_classes = [0, 1, 2, 3]
test_classes = [4, 5, 6, 7]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

criterion = CrossEntropyLoss()

can = Model(scale_cls=1, num_classes=50).to(device)

mcan = l2l.algorithms.GBML(
    module=can,
    transform=l2l.optim.MetaCurvatureTransform,
    lr=0.2,
    adapt_transform=False,
    first_order=True,  # has both 1st and 2nd order versions
).to(device)

slabel = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]]).to(device)
qlabel = torch.tensor([[0, 1, 2]]).to(device)

slabel_oh = one_hot(slabel).to(device)
qlabel_oh = one_hot(qlabel).to(device)


def aug_wav_2_mel(wav, augmentor):
    aug, _ = augmentor.filter_aug(wav)
    aug_mel_spec = librosa.feature.melspectrogram(
        y=aug.numpy(),
        sr=16000,
        n_fft=2048,
        hop_length=512,
        n_mels=128  # To avoid internal padding
    )
    return torch.tensor(
        np.log(1 + 10000 * aug_mel_spec[:, :int((16000 * 3) / 512)]).reshape(1, 128, int((16000 * 3) / 512)))


augmentor = RandAugment(aug_methods=['Noise', 'Filter', 'Pitch shift'],
                        aug_scale=4,
                        rir_df_path='',
                        sample_rate=16000)

# Define the optimizer
meta_opt = torch.optim.Adam(mcan.parameters(), lr=1e-3)

checkpoint_path = ''  # Update this path
checkpoint = torch.load(checkpoint_path)
mcan.load_state_dict(checkpoint['model_state_dict'])

val_eps = 5000
n_q = 10
steps = 8

PRE_VAL = []
POST_VAL = []

for episode in tqdm(range(val_eps), leave=True):
    val_model = mcan.clone()

    s_v, q_v, pids_v, wav_v = wav_episodic_sampling(3, 5, test_classes, dataset, n_query=n_q)
    s_v, q_v = s_v.cuda(), q_v.unsqueeze(0).cuda()

    val_model.eval()
    with torch.no_grad():
        for i in range(n_q):
            pre_cls_scores, _, _ = val_model(s_v.view(1, 15, 1, 128, 93), q_v[:, :, i], slabel_oh, qlabel_oh)
            _, pre_preds = torch.max(pre_cls_scores.view(3 * 1, -1).detach().cpu(), 1)
            pre_acc = (torch.sum(pre_preds == qlabel.detach().cpu()).float()) / qlabel.size(1)
            PRE_VAL.append(pre_acc.item())

    val_model.train()
    for step in range(steps):
        # ADFT
        for j in range(s_v.size(1)):
            s_f = torch.stack([
                torch.cat((
                    torch.cat((s_v[i, :j], s_v[i, j + 1:])),  # Removes the j-th column
                    (aug_wav_2_mel(torch.cat((wav_v[i, :j], wav_v[i, j + 1:]))[j - 1 if j > 0 else -1], augmentor))
                    .unsqueeze(0).to(device)
                ))
                for i in range(s_v.size(0))])
            q_f = torch.stack([s_v[i, j] for i in range(s_v.size(0))]).unsqueeze(0)

            ytest, cls_scores = val_model(s_f.view(1, 15, 1, 128, 93).to(device), q_f.to(device), slabel_oh, qlabel_oh)
            # loss1 = criterion(ytest, pids.view(-1))
            loss = criterion(cls_scores, qlabel.view(-1))
            # loss = loss1 + 0.5 * loss2
            val_model.adapt(loss, allow_unused=True)
            # LOSS.append(loss.item())

    val_model.eval()
    with torch.no_grad():
        for i in range(n_q):
            # print(q_v[:, i].unsqueeze(0).shape)
            post_cls_scores, _, _ = val_model(s_v.view(1, 15, 1, 128, 93), q_v[:, :, i], slabel_oh, qlabel_oh)
            _, post_preds = torch.max(post_cls_scores.view(3 * 1, -1).detach().cpu(), 1)
            post_acc = (torch.sum(post_preds == qlabel.detach().cpu()).float()) / qlabel.size(1)
            POST_VAL.append(post_acc.item())
    print(
        f'pre_val:{np.mean(PRE_VAL):.4f}, post_val: {np.mean(POST_VAL):.4f},')

print(mean_confidence_interval(PRE_VAL))
print(mean_confidence_interval(POST_VAL))
