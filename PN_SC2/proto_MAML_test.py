from proto_net import load_protonet_conv
from utils import episodic_sampling, audio_to_melspectrogram, mean_confidence_interval
import torch
import numpy as np
import os
import learn2learn as l2l
from tqdm import tqdm
import scipy.stats
from datasets import load_dataset
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dataset = load_dataset("speech_commands", "v0.02")
train = dataset['train']
val = dataset['validation']
test = dataset['test']

train = train.map(audio_to_melspectrogram, remove_columns=["audio"])
test = test.map(audio_to_melspectrogram, remove_columns=["audio"])
val = val.map(audio_to_melspectrogram, remove_columns=["audio"])

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

test_episodes = 50000
steps = 8

proto = load_protonet_conv([1, 128, 32], 128, 128)

maml = l2l.algorithms.GBML(
    module=proto,
    transform=l2l.optim.MetaCurvatureTransform,
    lr=0.02,
    adapt_transform=False,
    first_order=True
).to(device)

PRE_VAL = []
TRAIN_ACC = []
POST_VAL = []

checkpoint_path = ''
checkpoint = torch.load(checkpoint_path)
maml.load_state_dict(checkpoint['model_state_dict'])

for i in tqdm(range(test_episodes), leave=True):
    task_model = maml.clone()

    with torch.no_grad():
        s_v, q_v = episodic_sampling(5, 5, test_classes, test_dataset)
        s_v = s_v.to(device)
        q_v = q_v.unsqueeze(1).to(device)
        pre_loss, pre_pred, pre_acc = task_model.module.loss(s_v, q_v)
        PRE_VAL.append(pre_acc.item())
        # print(pre_pred)

        # for tr_batch in range(batches):
    for step in range(steps):
        # RDFT
        # for j in range(s_v.size(1)):
        #     s_f = torch.stack([torch.cat((s_v[i, :j], s_v[i, j + 1:])) for i in range(s_v.size(0))])
        #     q_f = torch.stack([s_v[i, j] for i in range(s_v.size(0))]).unsqueeze(1)

        # EDFT
        # for j in range(s_v.size(1)-1):
        #     s_f = torch.stack([s_v[i, :j + 1] for i in range(s_v.size(0))])
        #     q_f = torch.stack([s_v[i, j + 1] for i in range(s_v.size(0))]).unsqueeze(1)

        # ADFT (replication-only)
        for j in range(s_v.size(1)):
            s_f = torch.stack([
                torch.cat((
                    torch.cat((s_v[i, :j], s_v[i, j + 1:])),  # Removes the j-th column
                    torch.cat((s_v[i, :j], s_v[i, j + 1:]))[j - 1 if j > 0 else -1].unsqueeze(0)
                ))
                for i in range(s_v.size(0))])
            q_f = torch.stack([s_v[i, j] for i in range(s_v.size(0))]).unsqueeze(1)

            s_f = s_f.to(device)
            q_f = q_f.to(device)
            loss, pre, acc = task_model.module.loss(s_f, q_f)
            task_model.adapt(loss)
            TRAIN_ACC.append(acc.item())

    with torch.no_grad():
        post_loss, post_pred, post_acc = task_model.module.loss(s_v, q_v)
        # print(post_pred)
        POST_VAL.append(post_acc.item())

    print(
        f"Episode [{i + 1}/{test_episodes}], pre_val_acc: {np.mean(PRE_VAL):.4f}, train_acc: {np.mean(TRAIN_ACC):.4f}, "
        f"post_val_acc: {np.mean(POST_VAL):.4f}")

pre = mean_confidence_interval(PRE_VAL)
post = mean_confidence_interval(POST_VAL)
print(pre, post)
