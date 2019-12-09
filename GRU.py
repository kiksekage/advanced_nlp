import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import *
from embeddings import *

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers, drop):
        super(GRUEncoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=layers, dropout=drop)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded

        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.layers, 1, self.hidden_size, device=device)

class GRUDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, layers, drop):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=layers, dropout=drop)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.layers, 1, self.hidden_size, device=device)
