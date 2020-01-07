import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import *
from embeddings import *

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.0, layers=1, mode='RNN'):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.layers = layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.hidden_layer = nn.RNN(self.hidden_size, self.hidden_size, num_layers=self.layers)

        if mode == 'LSTM':
        	self.hidden_layer = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.layers)
        elif mode == 'GRU':
        	self.hidden_layer = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.layers)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = self.dropout(output)
        
        output, hidden = self.hidden_layer(output, hidden)
        return output, hidden

    def initHidden(self):
        hidden = torch.zeros(self.layers, 1, self.hidden_size, device=device)
        nn.init.xavier_uniform_(hidden, gain=nn.init.calculate_gain('relu'))
        return hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.0, layers=1, attention=False, mode='RNN'):
    	#layers should be either 1 or 2
    	#in the latter case remember to pass a pair of hidden states!
    	#mode can be either 'LSTM', 'GRU' or 'RNN'
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.layers = layers
        self.max_length = max_length
        self.attention = attention

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        if self.attention:
	        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
	        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.hidden_layer = nn.RNN(self.hidden_size, self.hidden_size, num_layers=self.layers)

        if mode == 'LSTM':
        	self.hidden_layer = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.layers)
        elif mode == 'GRU':
        	self.hidden_layer = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.layers)
        
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs=None):
    	#encoder_outputs needed only if attention enabled
        output = self.embedding(input).view(1, 1, -1)
        output = self.dropout(output)

        if self.attention:
	        attn_weights = F.softmax(
	            self.attn(torch.cat((output[0], hidden[0]), 1)), dim=1)
	        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
	                                 encoder_outputs.unsqueeze(0))
	        output = torch.cat((output[0], attn_applied[0]), 1)
        	output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.hidden_layer(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        hidden = torch.zeros(self.layers, 1, self.hidden_size, device=device)
        nn.init.xavier_uniform_(hidden, gain=nn.init.calculate_gain('relu'))
        return hidden
