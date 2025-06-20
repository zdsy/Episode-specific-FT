from proto_net import load_protonet_conv
from utils import episodic_sampling, audio_to_melspectrogram
import torch
import numpy as np
import os
import learn2learn as l2l
from tqdm import tqdm
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

epochs = 500
val_episodes = 5000
batch = 1
if batch == 1:
    episodes = 1000
else:
    episodes = 100
steps = 8
checkpoint_dir = 'sc2_checkpoints/5-way-5-shot/ADFT/random_mel'
os.makedirs(checkpoint_dir, exist_ok=True)

proto = load_protonet_conv([1, 128, 32], 128, 128)

# Meta-curvature
maml = l2l.algorithms.GBML(
    module=proto,
    transform=l2l.optim.MetaCurvatureTransform,
    lr=0.02,
    adapt_transform=False,
    first_order=True,  # has both 1st and 2nd order versions
).to(device)

meta_opt = torch.optim.Adam(maml.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=meta_opt, T_max=10, eta_min=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(meta_opt, step_size=80, gamma=0.6)

for epoch in range(epochs):

    PRE_VAL = []
    TRAIN_ACC = []
    POST_VAL = []

    for episode in tqdm(range(episodes), desc=f'Epoch {epoch + 1}/{epochs}', leave=True):

        meta_loss = 0
        meta_opt.zero_grad()

        for b in range(batch):
            task_model = maml.clone()

            with torch.no_grad():
                s_v, q_v = episodic_sampling(5, 5, train_classes, train_dataset)
                s_v = s_v.to(device)
                q_v = q_v.unsqueeze(1).to(device)
                # print(q_v.shape)
                pre_loss, pre_pred, pre_acc = task_model.module.loss(s_v, q_v)
                PRE_VAL.append(pre_acc.item())
                # print(pre_pred)

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

            post_loss, post_pred, post_acc = task_model.module.loss(s_v, q_v)
            # print(post_pred)
            POST_VAL.append(post_acc.item())
            meta_loss += post_loss
            # post_loss.backward()
            # meta_opt.step()

        meta_loss /= batch
        meta_loss.backward()
        meta_opt.step()

        maml.train()

    # print(f"Epoch [{epoch + 1}/{epochs}], Meta-Loss: {meta_loss.item():.4f}")
    print(f"Epoch [{epoch + 1}/{epochs}], pre_val_acc: {np.mean(PRE_VAL):.4f}, train_acc: {np.mean(TRAIN_ACC):.4f}, "
          f"post_val_acc: {np.mean(POST_VAL):.4f}")

    scheduler.step()

    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': maml.state_dict(),
        'optimizer_state_dict': meta_opt.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        # Add any other information you need

    }, checkpoint_path)

    # print(loss_record.index(min(loss_record)))

    # print(f'Model checkpoint saved to {checkpoint_path}')
    # manage_checkpoints(checkpoint_dir, max_checkpoints=3000)

    if ((epoch + 1) % 3) == 0:

        VAL_PRE_VAL = []
        VAL_TRAIN_ACC = []
        VAL_POST_VAL = []

        for i in tqdm(range(val_episodes), leave=True):
            val_model = maml.clone()

            with torch.no_grad():
                s_v, q_v = episodic_sampling(5, 5, val_classes, val_dataset)
                s_v = s_v.to(device)
                q_v = q_v.unsqueeze(1).to(device)
                pre_loss, pre_pred, pre_acc = val_model.module.loss(s_v, q_v)
                VAL_PRE_VAL.append(pre_acc.item())
                # print(pre_pred)

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
                    loss, pre, acc = val_model.module.loss(s_f, q_f)
                    val_model.adapt(loss)
                    VAL_TRAIN_ACC.append(acc.item())

            with torch.no_grad():
                post_loss, post_pred, post_acc = val_model.module.loss(s_v, q_v)
                # print(post_pred)
                VAL_POST_VAL.append(post_acc.item())

        print(
            f"VAL_epoch:{epoch + 1}, pre_val_acc: {np.mean(VAL_PRE_VAL):.4f}, train_acc: {np.mean(VAL_TRAIN_ACC):.4f}, "
            f"post_val_acc: {np.mean(VAL_POST_VAL):.4f}")
