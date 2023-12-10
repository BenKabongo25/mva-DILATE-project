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

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        

    def forward(self, input, h0):
        output, hn = self.rnn(input, h0)
        return output, hn


class Seq2SeqModel(nn.Module):

    def __init__(self, encoder, decoder, output_length) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
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
