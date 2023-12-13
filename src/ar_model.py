# Machine Learning for Times Series
# DILATE Project
#
# Ben Kabongo & Martin Brosset
# M2 MVA

# AR MPL Model


import torch
import torch.nn as nn
import torch.nn.functional as F


class ArMlpModel(nn.Module):

    def __init__(self, input_size, degree, device=None):
        super().__init__()
        self.input_size = input_size
        self.degree = degree
        self.fcs = nn.ModuleList([nn.Linear(input_size, input_size, device=device)] for _ in range(degree))
        self.out = nn.Linear(input_size, input_size, device=device)

    def forward(self, X):
        Z = torch.zeros((X.size(0), self.input_size))
        for i, fc in enumerate(self.fcs):
            Z += F.relu(fc(X[:, i, :]))
        Z = self.out(Z)
        return Z

    def predict(self, X, output_length):
        outputs = torch.zeros((output_length, X.size(0), self.input_size), device=self.device)
        X = X[:, -self.degree:, :]
        for i in range(output_length):
            Z = self.forward(X)
            output_length[i] = Z
            X = torch.cat((X[:, 1:, :], Z.unsqueeze(1)), dim=1)
        return torch.transpose(outputs, 0, 1)
        
