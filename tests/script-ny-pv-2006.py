import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import torch
import warnings

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


BASE_PATH = ".."

sys.path.append(BASE_PATH + "/src/")
from seq2seq import Seq2SeqModel
from trainer import repeat


warnings.filterwarnings(action="ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
plt.style.use("bmh")


# dataset link : https://www.nrel.gov/grid/assets/downloads/ny-pv-2006.zip
DATA_PATH = BASE_PATH + "/data/ny-pv-2006/"

class SolarDataset(Dataset):

    def __init__(self, data, output_length=100):
        self.data = torch.from_numpy(data).to(dtype=torch.float32)
        self.output_length = output_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, :-self.output_length], self.data[index, -self.output_length:]


times_series = {"Actual": {"UPV": [], "DPV": []}, "DA": {"UPV": [], "DPV": []}, "HA4": {"UPV": [], "DPV": []}}

for file in os.listdir(DATA_PATH)[:100]:
    x = pd.read_csv(DATA_PATH + file)[["Power(MW)"]].to_numpy()
    
    data_type = ""
    if file.startswith("Actual"):
        data_type = "Actual"
    elif file.startswith("DA"):
        data_type = "DA"
    elif file.startswith("HA4"):
        data_type = "HA4"
    
    pv_type = ""
    if "DPV" in file:
        pv_type = "DPV"
    else:
        pv_type = "UPV"
    
    times_series[data_type][pv_type].append(x.tolist())


batch_size = 32
output_length = 100
train_size = 0.4

dataloaders = {"Actual": {"UPV": [], "DPV": []}, "DA": {"UPV": [], "DPV": []}, "HA4": {"UPV": [], "DPV": []}}
for data_type in ["Actual", "DA", "HA4"]:
    for pv_type in ["DPV", "UPV"]:
        data = np.array(times_series[data_type][pv_type]).reshape((-1,  365, 1))
        times_series[data_type][pv_type] = data
        print(data_type, pv_type, times_series[data_type][pv_type].shape)

        data_train, data_test = train_test_split(data, train_size=train_size)
        train_dataset = SolarDataset(data_train, output_length=output_length)
        test_dataset = SolarDataset(data_test, output_length=output_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        dataloaders[data_type][pv_type] = {"train": train_dataloader, "test": test_dataloader}


        # TODO: Experimentations

