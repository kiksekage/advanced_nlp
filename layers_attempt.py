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
        self.gru = nn.GRU(hidden_size, hidden_size) # num_layers=2,

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        #CustomLoss.forward(input)

        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size) #num_layers=2
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#loss = CustomLoss.apply

def train(input_tensor, output_tensor, encoder, encoder_optimizer, decoder, decoder_optimizer, criterion, max_length=100, clipping_value=5):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    output_length = output_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    forcing = random.random()

    if forcing > 0.5:
        for di in range(output_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_input = output_tensor[di]
            loss += criterion(decoder_output, output_tensor[di])

            if decoder_input.item() == EOS_token:
                break
    else:
        for di in range(output_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, output_tensor[di])

            if decoder_input.item() == EOS_token:
                break

    #loss = CustomLoss
    loss.backward()

    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clipping_value)
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clipping_value)

    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / output_length

    
def trainIters(encoder, decoder, train_data, input_lang, output_lang, trials=10000, learning_rate=0.001):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    actual_trials = 0
    losses = []
    print(train_data.shape[0])
    print_loss_total = 0

    while actual_trials < trials:
        chosen_datapoint = random.choice(train_data)
        training_pair = tensorsFromPair(chosen_datapoint, input_lang, output_lang)
        input_tensor = training_pair[0]
        output_tensor = training_pair[1]

        loss = train(input_tensor, output_tensor, encoder, encoder_optimizer, decoder, decoder_optimizer, criterion)
        losses.append(loss)
        print_loss_total += loss

        if actual_trials % 100 == 0:
            print_loss_avg = print_loss_total / 100
            print(print_loss_avg)
            print_loss_total = 0
        
        actual_trials +=1

    return losses
