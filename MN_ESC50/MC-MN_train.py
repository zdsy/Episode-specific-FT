import os
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import learn2learn as l2l
from model import MatchingNetwork
from tqdm import tqdm
import random
import torch
import numpy as np
from data_loader import ESC50Dataset
from utils import episodic_sampling, one_hot

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

# Define the optimizer
optimizer = torch.optim.Adam(mcmn.parameters(), lr=0.001)

checkpoint_dir = 'MCMN_checkpoints/5-way-5-shot/ADFT/rand_mel'
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
num_epochs = 1000  # Define the number of epochs
episodes = 1000
val_eps = 500
batch_size = 1
# patience = 5
best_acc = 0
best_epoch = 0
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

# pseudo-FT-label
slabel_ft = one_hot(torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]))
qlabel_ft = torch.tensor([0, 1, 2, 3, 4])
slabel_ft_batch = []
qlabel_ft_batch = []
for b in range(batch_size):
    slabel_ft_batch.append(slabel_ft)
    qlabel_ft_batch.append(qlabel_ft)
slabel_ft_batch = torch.stack(slabel_ft_batch)
qlabel_ft_batch = torch.stack(qlabel_ft_batch)

for epoch in range(num_epochs):
    # LOSS = []
    PRE_ACC = []
    POST_ACC = []
    TR_ACC = []
    for episode in tqdm(range(episodes), desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True):
        # Clone MC model
        task_model = mcmn.clone()

        # Prepare batch data
        sbatch = []
        qbatch = []
        for b in range(batch_size):
            s, q = episodic_sampling(5, 5, train_classes, train_dataset)
            sbatch.append(s)
            qbatch.append(q)
        sbatch = torch.stack(sbatch)
        qbatch = torch.stack(qbatch)
        # print(sbatch.shape, qbatch.shape)

        # sbatch, qbatch = episodic_sampling(5, 5, train_classes, train_dataset)
        # sbatch = sbatch.to(device)
        # qbatch = qbatch.to(device)

        # Calculate Pre Acc
        with torch.no_grad():
            pre_acc, _ = task_model.module(sbatch.view(batch_size, 25, 1, 128, 160).cuda(), slabel_batch.cuda(),
                                           qbatch.view(batch_size, 5, 1, 128, 160).cuda(), qlabel_batch.cuda())
            PRE_ACC.append(pre_acc.item())

        for step in range(steps):
            # ADFT (replication-only)
            sbatch = sbatch.view(5, 5, 1, 128, 160)

            for j in range(sbatch.size(1)):
                s_f = torch.stack([
                    torch.cat((
                        torch.cat((sbatch[i, :j], sbatch[i, j + 1:])),  # Removes the j-th column
                        torch.cat((sbatch[i, :j], sbatch[i, j + 1:]))[j - 1 if j > 0 else -1].unsqueeze(0)
                    ))
                    for i in range(sbatch.size(0))])
                q_f = torch.stack([sbatch[i, j] for i in range(sbatch.size(0))])

                tr_acc, loss = task_model.module(s_f.view(batch_size, 25, 1, 128, 160).cuda(),
                                                 slabel_batch.cuda(),
                                                 q_f.view(batch_size, 5, 1, 128, 160).cuda(), qlabel_batch.cuda())

            # RDFT
            # for j in range(sbatch.size(2)):
            #     s_f = torch.stack([torch.cat((sbatch[:, i, :j], sbatch[:, i, j + 1:]), 1)  # Removes the j-th column
            #                         for i in range(sbatch.size(1))], 1)
            #     q_f = torch.stack([sbatch[:, i, j] for i in range(sbatch.size(1))], 1)

            #     tr_acc, loss = task_model.module(s_f.view(batch_size, 20, 1, 128, 160).cuda(),
            #                                      slabel_ft_batch.cuda(),
            #                                      q_f.view(batch_size, 5, 1, 128, 160).cuda(), qlabel_batch.cuda())

                task_model.adapt(loss)
                TR_ACC.append(tr_acc.item())

        # Calculate Post Acc
        post_acc, post_loss = task_model.module(sbatch.view(batch_size, 25, 1, 128, 160).cuda(), slabel_batch.cuda(),
                                                qbatch.view(batch_size, 5, 1, 128, 160).cuda(), qlabel_batch.cuda())
        POST_ACC.append(post_acc.item())

        # No accumulation on meta_loss, cause with the usage of batch, post_loss is already an averaged loss over eps
        optimizer.zero_grad()
        post_loss.backward()
        optimizer.step()

        # Print statistics
    print(f'Epoch [{epoch + 1}/{num_epochs}], pre_acc: {np.mean(PRE_ACC):.4f}, tr_acc: {np.mean(TR_ACC):.4f},'
          f'post_acc: {np.mean(POST_ACC):.4f}')

    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': mcmn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Add any other information you need

    }, checkpoint_path)

    # Validation episodes
    if (((epoch + 1) % 3) == 0) and (np.mean(POST_ACC) > 0.95):
        PRE_VAL_ACC = []
        POST_VAL_ACC = []
        TR_VAL_ACC = []
        for val_episode in tqdm(range(val_eps), desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True):
            # Clone MC model
            val_model = mcmn.clone()

            # Prepare batch data
            sbatch_val = []
            qbatch_val = []
            for b in range(batch_size):
                s, q = episodic_sampling(5, 5, val_classes, val_dataset)
                sbatch_val.append(s)
                qbatch_val.append(q)
            sbatch_val = torch.stack(sbatch_val)
            qbatch_val = torch.stack(qbatch_val)

            # Calculate Pre Acc
            with torch.no_grad():
                pre_val_acc, _ = val_model.module(sbatch_val.view(batch_size, 25, 1, 128, 160).cuda(),
                                                  slabel_batch.cuda(),
                                                  qbatch_val.view(batch_size, 5, 1, 128, 160).cuda(),
                                                  qlabel_batch.cuda())
                PRE_VAL_ACC.append(pre_val_acc.item())

            for step in range(steps):
                # ADFT (replication-only)
                sbatch_val = sbatch_val.view(5, 5, 1, 128, 160)

                for j in range(sbatch_val.size(1)):
                    s_f = torch.stack([
                        torch.cat((
                            torch.cat((sbatch_val[i, :j], sbatch_val[i, j + 1:])),  # Removes the j-th column
                            torch.cat((sbatch_val[i, :j], sbatch_val[i, j + 1:]))[j - 1 if j > 0 else -1].unsqueeze(0)
                        ))
                        for i in range(sbatch_val.size(0))])
                    q_f = torch.stack([sbatch_val[i, j] for i in range(sbatch_val.size(0))])

                    tr_val_acc, val_loss = val_model.module(s_f.view(batch_size, 25, 1, 128, 160).cuda(),
                                                            slabel_batch.cuda(),
                                                            q_f.view(batch_size, 5, 1, 128, 160).cuda(),
                                                            qlabel_batch.cuda())

                # RDFT
                # for j in range(sbatch_val.size(2)):
                #     s_f = torch.stack([torch.cat((sbatch_val[:, i, :j], sbatch_val[:, i, j + 1:]), 1)  # Removes the j-th column
                #                         for i in range(sbatch_val.size(1))], 1)
                #     q_f = torch.stack([sbatch_val[:, i, j] for i in range(sbatch_val.size(1))], 1)
                #
                #     tr_val_acc, val_loss = val_model.module(s_f.view(batch_size, 20, 1, 128, 160).cuda(),
                #                                     slabel_ft_batch.cuda(),
                #                                     q_f.view(batch_size, 5, 1, 128, 160).cuda(), qlabel_batch.cuda())

                    val_model.adapt(val_loss)
                    TR_VAL_ACC.append(tr_val_acc.item())

            # Calculate Post Acc
            post_val_acc, post_val_loss = val_model.module(sbatch_val.view(batch_size, 25, 1, 128, 160).cuda(),
                                                           slabel_batch.cuda(),
                                                           qbatch_val.view(batch_size, 5, 1, 128, 160).cuda(),
                                                           qlabel_batch.cuda())
            POST_VAL_ACC.append(post_val_acc.item())

        if np.mean(POST_VAL_ACC) > best_acc:
            best_acc = np.mean(POST_VAL_ACC)
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1

        if patience > 10:
            print('Training stopped due to no improvement for validation...')
            break

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], pre_val_acc: {np.mean(PRE_VAL_ACC):.4f}, '
            f'post_val_acc: {np.mean(POST_VAL_ACC):.4f}, best_val_acc: {best_acc:.4f}, best epoch: {best_epoch}')
