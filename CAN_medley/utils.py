import os
import pandas as pd
import torch
import librosa
import numpy as np
import random
import scipy

def adjust_learning_rate(optimizer, iters, LUT):
    # decay learning rate by 'gamma' for every 'stepsize'
    for (stepvalue, base_lr) in LUT:
        if iters < stepvalue:
            lr = base_lr
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h

def one_hot(labels_train):
    """
    Turn the labels_train to one-hot encoding.
    Args:
        labels_train: [batch_size, num_train_examples]
    Return:
        labels_train_1hot: [batch_size, num_train_examples, K]
    """
    labels_train = labels_train.cpu()
    nKnovel = 1 + labels_train.max()
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
    return labels_train_1hot

def episodic_sampling(c_way, k_shot, base_classes, data, n_query=1):
    ways = random.sample(list(base_classes), c_way)
    support_set = []
    query_set = []
    for way in ways:
        indicies = data.meta_data[data.meta_data['instrument_id'] == way].index.tolist()
        indicies = random.sample(indicies, k_shot + n_query)
        if k_shot == 1:
            support = data[indicies[0]][0]
        else:
            support = torch.stack([data[i][0] for i in indicies[:-n_query]])
        if n_query == 1:
            query = data[indicies[-1]][0]
        else:
            query = torch.stack([data[i][0] for i in indicies[-n_query:]])
        support_set.append(support)
        query_set.append(query)

    return torch.stack(support_set), torch.stack(query_set), torch.tensor(ways).unsqueeze(0)

def wav_episodic_sampling(c_way, k_shot, base_classes, data, n_query=1):
    ways = random.sample(list(base_classes), c_way)
    support_set = []
    query_set = []
    wav_set = []
    for way in ways:
        indicies = data.meta_data[data.meta_data['instrument_id'] == way].index.tolist()
        indicies = random.sample(indicies, k_shot + n_query)

        support = torch.stack([data[i][0] for i in indicies[:-n_query]])
        wav = torch.stack([data[i][2] for i in indicies[:-n_query]])

        if n_query == 1:
            query = data[indicies[-1]][0]
        else:
            query = torch.stack([data[i][0] for i in indicies[-n_query:]])

        support_set.append(support)
        query_set.append(query)
        wav_set.append(wav)

    return torch.stack(support_set), torch.stack(query_set), torch.tensor(ways).unsqueeze(0), torch.stack(wav_set)