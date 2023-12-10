# Machine Learning for Times Series
# DILATE Project
#
# Ben Kabongo & Martin Brosset
# M2 MVA


import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, input):
        output, hn = self.rnn(input)
        return output, hn


class Decoder(nn.Module):

    def __init__(self, n_dim, input_size, hidden_size, projection_size, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.project_fc = nn.Linear(hidden_size, projection_size)
        self.out_fc = nn.Linear(projection_size, n_dim)
        self.relu = nn.ReLU()
        
    def forward(self, input, h0):
        output, hn = self.rnn(input, h0)
        output = self.out_fc(self.relu(self.project_fc(output)))
        return output, hn


class Seq2SeqModel(nn.Module):

    def __init__(self, n_dim, output_length, input_size, hidden_size, projection_size, num_layers):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(n_dim, input_size, hidden_size, projection_size, num_layers)
        self.output_length = output_length

    def forward(self, X):
        outputs = torch.zeros((X.size(0), self.output_length, X.size(2))).to(self.device)
        input = X[:, -1, :]
        _, h = self.encoder(X)
        for j in range(self.output_length):
            output, h = self.decoder(input, h)
            outputs[:, j] = output
            input = output
        return outputs
