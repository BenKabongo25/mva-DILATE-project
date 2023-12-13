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

#BASE_PATH = "mva-DILATE-project"
BASE_PATH = ".."

sys.path.append(BASE_PATH + "/src/")
from seq2seq import Seq2SeqModel
from trainer import repeat


plt.style.use("bmh")
warnings.filterwarnings(action="ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


DATA_PATH = BASE_PATH + "/data/"
output_length = 56

ecg_train = np.array(pd.read_table(DATA_PATH + "ECG5000/ECG5000_TRAIN.tsv"))[:, :, np.newaxis]
ecg_test = np.array(pd.read_table(DATA_PATH + "ECG5000/ECG5000_TEST.tsv"))[:, :, np.newaxis]
print(ecg_train.shape, ecg_test.shape)


class ECG5000Dataset(Dataset):

    def __init__(self, data, output_length=output_length):
        self.data = torch.from_numpy(data).to(dtype=torch.float32)
        self.output_length = output_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, :-self.output_length], self.data[index, -self.output_length:]


batch_size = 32
ecg_train_dataset = ECG5000Dataset(ecg_train)
ecg_test_dataset = ECG5000Dataset(ecg_test)
train_dataloader = DataLoader(ecg_train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(ecg_test_dataset, batch_size=batch_size, shuffle=False)


n_runs = 10
lr = 1e-2
alpha = 0.5
gamma = 1e-2
n_epochs = 1_000


logs_repeat_path = BASE_PATH + "/logs/ecg5000_alpha_05_gamma_001/"
os.makedirs(logs_repeat_path, exist_ok=True)


model = Seq2SeqModel(output_length=output_length, input_size=1, hidden_size=128, projection_size=16, num_layers=1, device=device)

print("MSE ...")
results_MSE = repeat(n_runs, model, train_dataloader, test_dataloader, lr, n_epochs, alpha, gamma,
                     train_loss="MSE", device=device, path=logs_repeat_path)
print(results_MSE)

results_DILATE = repeat(n_runs, model, train_dataloader, test_dataloader, lr, n_epochs, alpha, gamma,
                        train_loss="DILATE", device=device, path=logs_repeat_path)
print(results_DILATE)
