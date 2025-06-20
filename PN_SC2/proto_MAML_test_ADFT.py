from proto_net import load_protonet_conv
from utils import wav_episodic_sampling, mean_confidence_interval, audio_to_melspectrogram
import torch
import pandas as pd
import numpy as np
import os
import learn2learn as l2l
from tqdm import tqdm
from aug import RandAugment
import librosa
from datasets import load_dataset
import torchaudio.transforms as T

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dataset = load_dataset("speech_commands", "v0.02")
train = dataset['train']
val = dataset['validation']
test = dataset['test']

train = train.map(audio_to_melspectrogram)
test = test.map(audio_to_melspectrogram)
val = val.map(audio_to_melspectrogram)

test = pd.DataFrame(test)
val = pd.DataFrame(val)
train = pd.DataFrame(train)

data = pd.concat([train, val, test], ignore_index=True)

np.random.seed(99)
classes = np.arange(35)
np.random.shuffle(classes)
train_classes = classes[:20]
val_classes = classes[20:27]
test_classes = classes[27:35]

train_dataset = data[data.label.isin(train_classes)].reset_index(drop=True)
test_dataset = data[data.label.isin(test_classes)].reset_index(drop=True)
val_dataset = data[data.label.isin(val_classes)].reset_index(drop=True)

print(len(train), len(test))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

augmentor = RandAugment(aug_methods=['Noise', 'Filter', 'Pitch shift'],
                        aug_scale=1,
                        rir_df_path='',
                        sample_rate=16000)


def aug_wav_2_mel(wav, augmentor):
    aug, _ = augmentor.filter_aug(wav)
    transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    mel = transform(aug.float())
    return T.AmplitudeToDB()(mel)


episodes = 50000

steps = 8

proto = load_protonet_conv([1, 128, 32], 128, 128)

# Meta-curvature
maml = l2l.algorithms.GBML(
    module=proto,
    transform=l2l.optim.MetaCurvatureTransform,
    lr=0.02,
    adapt_transform=False,
    first_order=True,  # has both 1st and 2nd order versions
).to(device)

checkpoint_path = ''
checkpoint = torch.load(checkpoint_path)
maml.load_state_dict(checkpoint['model_state_dict'])

PRE_VAL = []
TRAIN_ACC = []
POST_VAL = []

for episode in tqdm(range(episodes), leave=True):

    task_model = maml.clone()

    with torch.no_grad():
        s_v, q_v, wav = wav_episodic_sampling(5, 5, test_classes, test_dataset)
        s_v = s_v.to(device)
        q_v = q_v.unsqueeze(1).to(device)
        pre_loss, pre_pred, pre_acc = task_model.module.loss(s_v, q_v)
        PRE_VAL.append(pre_acc.item())
        # print(pre_pred)

    for step in range(steps):
        # ADFT
        for j in range(s_v.size(1)):
            s_f = torch.stack([
                        torch.cat((
                            torch.cat((s_v[i, :j], s_v[i, j + 1:])),  # Removes the j-th column
                            (aug_wav_2_mel(torch.cat((wav[i, :j], wav[i, j + 1:]))[j - 1 if j > 0 else -1], augmentor))
                            .unsqueeze(0).unsqueeze(0).to(device)
                        ))
                        for i in range(s_v.size(0))])
            q_f = torch.stack([s_v[i, j] for i in range(s_v.size(0))]).unsqueeze(1)

            s_f = s_f.to(device)
            q_f = q_f.to(device)
            loss, pre, acc = task_model.module.loss(s_f, q_f)
            task_model.adapt(loss)
            TRAIN_ACC.append(acc.item())

    post_loss, post_pred, post_acc = task_model.module.loss(s_v, q_v)
    # print(post_pred)
    POST_VAL.append(post_acc.item())

    print(
        f"Episode [{episode + 1}/{episodes}], pre_val_acc: {np.mean(PRE_VAL):.4f}, train_acc: {np.mean(TRAIN_ACC):.4f}, "
        f"post_val_acc: {np.mean(POST_VAL):.4f}")

pre = mean_confidence_interval(PRE_VAL)
post = mean_confidence_interval(POST_VAL)
print(pre, post)
