import os
import pandas as pd
import torch
from utils import adjust_learning_rate, episodic_sampling, one_hot, mean_confidence_interval
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

base_path = '../Medley-solo'
metadata_file = os.path.join(base_path, 'metadata.csv')

dataset = MedleySolosDBDataset(base_path, metadata_file)

train_classes = [0,1,2,3]
test_classes = [4,5,6,7]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

criterion = CrossEntropyLoss()

can = Model(scale_cls=1, num_classes=8).to(device)

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

checkpoint_path = ''  # Update this path
checkpoint = torch.load(checkpoint_path)
mcan.load_state_dict(checkpoint['model_state_dict'])

val_eps = 5000

# n_query = 10
steps = 8
n_q = 10

PRE_VAL = []
POST_VAL = []

for episode in tqdm(range(val_eps), leave=True):
    val_model = mcan.clone()

    s, q, pids = episodic_sampling(3, 5, test_classes, dataset, n_query=n_q)
    s, q = s.cuda(), q.cuda()

    # print(q.shape)

    val_model.eval()
    for i in range(n_q):
        with torch.no_grad():
            pre_cls_scores, _, _ = val_model(s.view(1, 15, 1, 128, 93), q[:, i].unsqueeze(0), slabel_oh, qlabel_oh)
            _, pre_preds = torch.max(pre_cls_scores.view(3 * 1, -1).detach().cpu(), 1)
            pre_acc = (torch.sum(pre_preds == qlabel.detach().cpu()).float()) / qlabel.size(1)
            PRE_VAL.append(pre_acc.item())

    val_model.train()
    for step in range(steps):
        # RDFT
        # for j in range(s.size(1)):
        #     s_f = torch.stack([torch.cat((s[i, :j], s[i, j + 1:])) for i in range(s.size(0))])
        #     q_f = torch.stack([s[i, j] for i in range(s.size(0))]).unsqueeze(0)

        #     slabel_f = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]]).to(device)
        #     slabel_f = one_hot(slabel_f).to(device)

        #     ytest, cls_scores = val_model(s_f.view(1, 12, 1, 128, 93).to(device), q_f.to(device), slabel_f,
        #                               qlabel_oh)
        #     # loss1 = criterion(ytest, pids.view(-1))
        #     loss = criterion(cls_scores, qlabel.view(-1))
        #     # loss = loss1 + 0.5 * loss2
        #     val_model.adapt(loss, allow_unused=True)
        #     # LOSS.append(loss.item())

        # EDFT
        # for j in range(s.size(1) - 1):
        #     s_f = torch.stack([s[i, :j + 1] for i in range(s.size(0))])
        #     q_f = torch.stack([s[i, j + 1] for i in range(s.size(0))]).unsqueeze(0)

        #     slabel_f = torch.tensor([[0] * (j + 1) + [1] * (j + 1) + [2] * (j + 1)]).to(device)
        #     slabel_f = one_hot(slabel_f).to(device)

        #     ytest, cls_scores = val_model(s_f.view(1, 3 * (j + 1), 1, 128, 93).to(device), q_f.to(device), slabel_f, qlabel_oh)
        #     # loss1 = criterion(ytest, pids.view(-1))
        #     loss = criterion(cls_scores, qlabel.view(-1))
        #     # loss = loss1 + 0.5 * loss2
        #     val_model.adapt(loss, allow_unused=True)
        #     # LOSS.append(loss.item()

        # ADFT (replication-only)
        for j in range(s.size(1)):
            s_f = torch.stack([
                torch.cat((
                    torch.cat((s[i, :j], s[i, j + 1:])),  # Removes the j-th column
                    torch.cat((s[i, :j], s[i, j + 1:]))[j - 1 if j > 0 else -1].unsqueeze(0)
                ))
                for i in range(s.size(0))])
            q_f = torch.stack([s[i, j] for i in range(s.size(0))]).unsqueeze(0)

            ytest, cls_scores = val_model(s_f.view(1, 15, 1, 128, 93).to(device), q_f.to(device), slabel_oh,
                                      qlabel_oh)
            # loss1 = criterion(ytest, pids.view(-1))
            loss = criterion(cls_scores, qlabel.view(-1))
            # loss = loss1 + 0.5 * loss2
            val_model.adapt(loss, allow_unused=True)
            # LOSS.append(loss.item())

    val_model.eval()
    for i in range(n_q):
        with torch.no_grad():
            post_cls_scores, _, _ = val_model(s.view(1, 15, 1, 128, 93), q[:, i].unsqueeze(0), slabel_oh, qlabel_oh)
            _, post_preds = torch.max(post_cls_scores.view(3 * 1, -1).detach().cpu(), 1)
            post_acc = (torch.sum(post_preds == qlabel.detach().cpu()).float()) / qlabel.size(1)
            POST_VAL.append(post_acc.item())

    print(f'Episode [{episode + 1}/{val_eps}], pre_val:{np.mean(PRE_VAL):.4f}, post_val: {np.mean(POST_VAL):.4f}')

pre = mean_confidence_interval(PRE_VAL)
post = mean_confidence_interval(POST_VAL)
print(pre, post)
