import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import os

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append("../src/")
from seq2seq import Seq2SeqModel
from dilate import DilateLoss

plt.style.use("bmh")
warnings.filterwarnings(action="ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def train(model, train_dataloader, test_dataloader, lr, n_epochs=1_000, alpha=0.01, gamma=0.01, train_loss="MSE"):

    all_train_loss = []
    shape_train_loss = []
    temporal_train_loss = []

    mse_test_loss = []
    dilate_test_loss = []
    shape_test_loss = []
    temporal_test_loss = []

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    mse_loss = torch.nn.MSELoss()
    dilate_loss = DilateLoss(alpha=alpha, gamma=gamma, device=device)

    tqdm_bar = tqdm(range(1, 1 + n_epochs), "Training")
    
    for epoch in tqdm_bar:
        model.train()
        epoch_train_loss = 0
        epoch_shape_train_loss = 0
        epoch_temporal_train_loss = 0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if train_loss == "MSE":
                loss = mse_loss(targets, outputs)
            else:
                loss, shape_loss, temporal_loss = dilate_loss(targets, outputs)
                epoch_shape_train_loss += shape_loss.item()
                epoch_temporal_train_loss += temporal_loss.item()

            epoch_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_train_loss.append(epoch_train_loss)
        if train_loss == "DILATE":
            shape_train_loss.append(epoch_shape_train_loss)
            temporal_train_loss.append(epoch_temporal_train_loss)

        model.eval()
        epoch_mse_test_loss = 0
        epoch_dilate_test_loss = 0
        epoch_shape_test_loss = 0
        epoch_temporal_test_loss = 0

        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                loss = mse_loss(targets, outputs)
                epoch_mse_test_loss += loss.item()
                
                loss, shape_loss, temporal_loss = dilate_loss(targets, outputs)
                epoch_dilate_test_loss += loss.item()
                epoch_shape_test_loss += shape_loss.item()
                epoch_temporal_test_loss += temporal_loss.item()
                
        mse_test_loss.append(epoch_mse_test_loss)
        
        dilate_test_loss.append(epoch_dilate_test_loss)
        shape_test_loss.append(epoch_shape_test_loss)
        temporal_test_loss.append(epoch_temporal_test_loss)

        description = f"[Epoch {epoch}/{n_epochs}] loss : train = {epoch_train_loss:.2f}"
        if train_loss == "DILATE":
            description += f" shape = {epoch_shape_train_loss:.2f} temp = {epoch_temporal_train_loss:.2f}"
        
        description += f" mse test = {epoch_mse_test_loss:.2f}"
        description += f" dilate test = {epoch_dilate_test_loss:.2f} shape = {epoch_shape_test_loss:.2f} temp = {epoch_temporal_test_loss:.2f}"

        tqdm_bar.set_description(description)

    return all_train_loss, mse_test_loss, dilate_test_loss, shape_train_loss, temporal_train_loss, shape_test_loss, temporal_test_loss


DATA_PATH = "../data/"

ecg_train = np.array(pd.read_table(DATA_PATH + "ECG5000/ECG5000_TRAIN.tsv"))[:, :, np.newaxis]
ecg_test = np.array(pd.read_table(DATA_PATH + "ECG5000/ECG5000_TEST.tsv"))[:, :, np.newaxis]

print(ecg_train.shape, ecg_test.shape)

class ECG5000Dataset(Dataset):

    def __init__(self, data, output_length=56):
        self.data = torch.from_numpy(data).to(dtype=torch.float32)
        self.output_length = output_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, :-self.output_length], self.data[index, -self.output_length:]
    
batch_size = 32
ecg_train_dataset = ECG5000Dataset(ecg_train)
ecg_test_dataset = ECG5000Dataset(ecg_test)
ecg_train_dataloader = DataLoader(ecg_train_dataset, batch_size=batch_size, shuffle=True)
ecg_test_dataloader = DataLoader(ecg_test_dataset, batch_size=batch_size, shuffle=False)

input, output = ecg_train_dataset[333]
input, output = input.numpy(), output.numpy()

lr = 1e-3
n_epochs = 200
model = Seq2SeqModel(output_length=56, input_size=1, hidden_size=128, projection_size=16, num_layers=1, device=device)
all_train_loss, mse_test_loss, dilate_test_loss, shape_train_loss, temporal_train_loss, shape_test_loss, temporal_test_loss = train(
    model, ecg_train_dataloader, ecg_test_dataloader, lr, n_epochs, alpha=0.01, gamma=0.01, train_loss="DILATE"
    )

# Create a directory 'plots' if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

plt.figure(figsize=(8, 5))
plt.plot(all_train_loss, label="Train loss")
plt.plot(dilate_test_loss, label="Test loss")
plt.legend()
plt.savefig('plots/train_test_dilate.png')  # Save figure
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(shape_train_loss, label="Train shape loss")
plt.plot(shape_test_loss, label="Test shape loss")
plt.legend()
plt.savefig('plots/test_dilate_mse.png')  # Save figure
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(temporal_train_loss, label="Train temporal loss")
plt.plot(temporal_test_loss, label="Test temporal loss")
plt.legend()
plt.savefig('plots/test_dilate_mse.png')  # Save figure
plt.close()

model.eval()
input, target = ecg_test_dataset[3333]
input = input.to(device)  # Move input to the same device as the model
prediction = model(input.unsqueeze(0)).squeeze()
input, target, prediction = input.numpy(), target.numpy(), prediction.detach().numpy()
plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, len(input)), input, label="input")
plt.plot(np.arange(len(input), len(input) + len(target)), target, label="target")
plt.plot(np.arange(len(input), len(input) + len(target)), prediction, label="prediction")
plt.legend()
plt.savefig('plots/reconstruction_ecg_3333.png')  # Save figure
plt.close()