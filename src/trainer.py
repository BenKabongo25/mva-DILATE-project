# Machine Learning for Times Series
# DILATE Project
#
# Ben Kabongo & Martin Brosset
# M2 MVA


import copy
import json
import os
import torch
from dilate import DilateLoss
from tqdm import tqdm
from tslearn.metrics import dtw_path


def train(model, train_dataloader, lr, n_epochs=1_000, alpha=0.01, gamma=0.01, train_loss="MSE", device=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = torch.nn.MSELoss()
    dilate_loss_fn = DilateLoss(alpha=alpha, gamma=gamma, device=device)

    tqdm_bar = tqdm(range(1, 1 + n_epochs), "Training")
    
    for epoch in tqdm_bar:
        model.train()
        total_loss = 0
        shape_loss = 0
        temporal_loss = 0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if train_loss == "MSE":
                loss = mse_loss_fn(targets, outputs)
            else:
                loss, shape_loss_, temporal_loss_ = dilate_loss_fn(targets, outputs)
                shape_loss += shape_loss_.item()
                temporal_loss += temporal_loss_.item()

            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        description = f"[Epoch {epoch}/{n_epochs}] loss : train = {total_loss:.2f}"
        if train_loss == "DILATE":
            description += f" shape = {shape_loss:.2f} temp = {temporal_loss:.2f}"

        tqdm_bar.set_description(description)


def evaluate(model, test_dataloader, alpha=0.01, gamma=0.01, device=None):
    mse_loss_fn = torch.nn.MSELoss()
    dilate_loss_fn = DilateLoss(alpha=alpha, gamma=gamma, device=device)

    model.eval()
    mse_loss = 0
    dilate_loss = 0
    shape_loss = 0
    temporal_loss = 0
    dtw_loss = 0
    tdi_loss = 0

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = mse_loss_fn(targets, outputs)
            mse_loss += loss.item()

            loss, shape_loss_, temporal_loss_ = dilate_loss_fn(targets, outputs)
            dilate_loss += loss.item()
            shape_loss += shape_loss_.item()
            temporal_loss += temporal_loss_.item()

            batch_size = inputs.size(0)
            output_length = targets.size(1)

            batch_dtw_loss = 0
            batch_tdi_loss = 0

            for i in range(batch_size):         
                path, sim = dtw_path(targets[i].flatten().detach().cpu().numpy(), 
                                     outputs[i].flatten().detach().cpu().numpy())   
                batch_dtw_loss += sim        
                for i, j in path:
                    batch_tdi_loss += (i - j) ** 2 / (output_length ** 2)
                              
            dtw_loss += batch_dtw_loss / batch_size
            tdi_loss += batch_tdi_loss / batch_size
        
    return {"mse": mse_loss, "dilate": dilate_loss, "shape": shape_loss, "temporal": temporal_loss, "dtw": dtw_loss, "tdi": tdi_loss}


def repeat(n_runs, model, train_dataloader, test_dataloader, 
          lr, n_epochs=1_000, alpha=0.01, gamma=0.01, train_loss="MSE", device=None, path=""):

    path = path + train_loss + ".json"
    all_results = {"mse": [], "dilate": [], "shape": [], "temporal": [], "dtw": [], "tdi": []}
    for n in range(n_runs):
        torch.random.manual_seed(n)
        new_model = copy.deepcopy(model)
        train(new_model, train_dataloader, lr, n_epochs, alpha, gamma, train_loss, device)
        run_results = evaluate(new_model, test_dataloader, alpha, gamma, device)
        for loss_name, loss_value in run_results.items():
            all_results[loss_name].append(loss_value)
        with open(path, "w") as f:
            f.write(json.dumps(all_results))
        
    return all_results


if __name__ == "__main__":

    from utils import generate_synthetic_dataset
    from seq2seq import Seq2SeqModel
    from torch.utils.data import Dataset, DataLoader
    
    class SyntheticDataset(Dataset):
        def __init__(self, inputs, outputs):
            super().__init__()
            self.inputs = torch.from_numpy(inputs).to(dtype=torch.float32)
            self.outputs = torch.from_numpy(outputs).to(dtype=torch.float32)

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, index):
            return self.inputs[index], self.outputs[index]

    n_samples = 32
    batch_size = 16
    n_dim = 1
    input_length = 10
    output_length = 10
    scale = 0.01
    n_breakpoints = 1

    inputs, outputs = generate_synthetic_dataset(n_samples, n_dim, input_length, output_length, scale, n_breakpoints)
    train_dataset = SyntheticDataset(inputs, outputs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    inputs, outputs = generate_synthetic_dataset(n_samples, n_dim, input_length, output_length, scale, n_breakpoints)
    test_dataset = SyntheticDataset(inputs, outputs)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Seq2SeqModel(output_length=output_length, input_size=n_dim, hidden_size=16, projection_size=8, num_layers=1, device=None)

    n_runs = 3
    lr = 1e-2
    alpha = 0.5
    gamma = 1e-2
    n_epochs = 2

    logs_repeat_path = "logs/test/"
    os.makedirs(logs_repeat_path, exist_ok=True)

    print("MSE ...")
    results_MSE = repeat(n_runs, model, train_dataloader, test_dataloader, lr, n_epochs, alpha, gamma, 
                        train_loss="MSE", path=logs_repeat_path)
    print(results_MSE)

    print("DILATE ...")
    results_DILATE = repeat(n_runs, model, train_dataloader, test_dataloader, lr, n_epochs, alpha, gamma, 
                            train_loss="DILATE", path=logs_repeat_path)
    print(results_DILATE)

