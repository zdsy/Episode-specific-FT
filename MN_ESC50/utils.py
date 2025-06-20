import random
import torch
import os
import librosa
import numpy as np
import scipy


def episodic_sampling(c_way, k_shot, base_classes, data, n_query=1):
    ways = random.sample(base_classes.tolist(), c_way)
    support_set = []
    query_set = []
    for way in ways:
        indicies = data.meta_data[data.meta_data['target'] == way].index.tolist()
        indicies = random.sample(indicies, k_shot + n_query)
        if k_shot == 1:
            support = data[indicies[0]][0]
        else:
            support = torch.stack([data[i][0] for i in indicies[:-n_query]])
        query = torch.stack([data[i][0] for i in indicies[-n_query:]])
        support_set.append(support)
        query_set.append(query)
    if n_query == 1:
        return torch.stack(support_set), torch.stack(query_set)
    else:
        return torch.stack(support_set), torch.stack(query_set).squeeze(0)


def one_hot(labels_train):
    labels_train = labels_train.cpu()
    nKnovel = 5
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel, ]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1,
                                                                     labels_train_unsqueeze, 1)
    return labels_train_1hot


def wav_episodic_sampling(c_way, k_shot, base_classes, data, n_query=1):
    ways = random.sample(base_classes.tolist(), c_way)
    support_set = []
    query_set = []
    wav_set = []
    for way in ways:
        indicies = data.meta_data[data.meta_data['target'] == way].index.tolist()
        indicies = random.sample(indicies, k_shot + n_query)
        if k_shot == 1:
            support = data[indicies[0]][0]
            wav = data[indicies[0]][2]
        else:
            support = torch.stack([data[i][0] for i in indicies[:-n_query]])
            wav = torch.stack([data[i][2] for i in indicies[:-n_query]])
        query = torch.stack([data[i][0] for i in indicies[-n_query:]])
        support_set.append(support)
        query_set.append(query)
        wav_set.append(wav)
    if n_query == 1:
        return torch.stack(support_set), torch.stack(query_set), torch.stack(wav_set)
    else:
        return torch.stack(support_set), torch.stack(query_set).squeeze(0), torch.stack(wav_set)


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
