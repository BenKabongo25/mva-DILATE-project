# Machine Learning for Times Series
# DILATE Project
#
# Ben Kabongo & Martin Brosset
# M2 MVA

# Fully connected network model (MLP)


import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):

    def __init__(self, sample_input_size, sequence_length):
        self.hiddens = []
        input_size = sample_input_size * sequence_length/2
        while input_size > sample_input_size:
            self.hiddens.append(nn.Linear(2 * input_size, input_size))
            input_size /= 2
        self.out_fc = nn.Linear(input_size, sample_input_size)
        self.sample_input_size = sample_input_size
        self.sequence_length = sequence_length

    def forward(self, X):
        out = X.reshape(X.size(0), -1)
        for fc in self.hiddens:
            out = F.relu(fc(out))
        out = self.out_fc(out)
