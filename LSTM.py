import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import *
from embeddings import *

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,num_layers=2) # num_layers=2,

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded

        hidden1, hidden2 = hidden
        output, (hidden1, hidden2) = self.lstm(output, (hidden1, hidden2))
        return output, (hidden1, hidden2)

    def initHidden(self):
        hidden = torch.zeros(2, 1, self.hidden_size, device=device)
        nn.init.xavier_uniform_(hidden, gain=nn.init.calculate_gain('relu'))
        return hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,num_layers=2) #num_layers=2
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        hidden1, hidden2 = hidden
        output, (hidden1, hidden2) = self.lstm(output, (hidden1, hidden2))
        output = self.softmax(self.out(output[0]))
        return output, (hidden1, hidden2)

    def initHidden(self):
        hidden = torch.zeros(2, 1, self.hidden_size, device=device)
        nn.init.xavier_uniform_(hidden, gain=nn.init.calculate_gain('relu'))
        return hidden
