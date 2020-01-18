import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import *
from embeddings import *
from enc_and_dec import *

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(input_tensor, output_tensor, encoder, encoder_optimizer, decoder, decoder_optimizer, criterion, max_length, clipping_value=5):
    encoder_hidden1 = encoder.initHidden()
    encoder_hidden2 = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    output_length = output_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        if encoder.mode == 'LSTM':
            encoder_output, (encoder_hidden1, encoder_hidden2) = encoder(input_tensor[ei], (encoder_hidden1, encoder_hidden2))
        else: 
            encoder_output, encoder_hidden1 = encoder(input_tensor[ei], encoder_hidden1)
        encoder_outputs[ei] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden1 = encoder_hidden1
    decoder_hidden2 = encoder_hidden2

    forcing = random.random() > 0.5

    if forcing:
        for di in range(output_length):
            if decoder.mode == 'LSTM':
              decoder_output, (decoder_hidden1, decoder_hidden2) = decoder(decoder_input, (decoder_hidden1, decoder_hidden2), encoder_outputs)
            else:
              decoder_output, decoder_hidden1 = decoder(decoder_input, decoder_hidden1, encoder_outputs)
            
            decoder_input = output_tensor[di]
            loss += criterion(decoder_output, output_tensor[di])

            if decoder_input.item() == EOS_token:
              break
    else:
        for di in range(output_length):
            if decoder.mode == 'LSTM':
              decoder_output, (decoder_hidden1, decoder_hidden2) = decoder(decoder_input, (decoder_hidden1, decoder_hidden2), encoder_outputs)
            else:
              decoder_output, decoder_hidden1 = decoder(decoder_input, decoder_hidden1, encoder_outputs)
            
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, output_tensor[di])
            
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clipping_value)
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clipping_value)

    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / output_length

    
def trainIters(encoder, decoder, train_data, input_lang, output_lang, max_length, learning_rate=0.001):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    losses = []
    print(train_data.shape[0])
    print_loss_total = 0

    for iter in range(train_data.shape[0]):
        training_pair = tensorsFromPair(train_data[iter], input_lang, output_lang)
        input_tensor = training_pair[0]
        output_tensor = training_pair[1]

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            output_tensor = output_tensor.cuda()
        
        loss = train(input_tensor, output_tensor, encoder, encoder_optimizer, decoder, decoder_optimizer, criterion, max_length)
        print_loss_total += loss

        if iter % 1000 == 0:
            print_loss_avg = print_loss_total / 500
            losses.append(print_loss_avg)
            print(iter)
            print(print_loss_avg)
            print_loss_total = 0

    return losses

def train_and_save(train_data, train_in, train_out, model, dropout, att, layers, model_name, file_prefix, hidden_units=200, MAX_LENGTH=100):
    encoder = Encoder(train_in.n_words, hidden_units, layers=layers, mode=model, dropout_p=dropout)
    decoder = Decoder(hidden_units, train_out.n_words, layers=layers, max_length=MAX_LENGTH, mode=model, dropout_p=dropout, attention=att)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    losses = trainIters(encoder, decoder, train_data, train_in, train_out, MAX_LENGTH)
    plt.plot(losses)
    plt.title(model+'_layers='+str(layers)+'_drop='+str(dropout)+'_attention='+str(att))
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()
    torch.save(encoder.state_dict(), file_prefix+model_name+"_encoder.pt")
    torch.save(decoder.state_dict(), file_prefix+model_name+"_decoder.pt")

def load_models(train_in_nwords, train_out_nwords, hidden_size, layers, mode, dropout_p, attention, file_location, model_name, max_length=100):
    encoder = Encoder(train_in_nwords, hidden_size, layers=layers, mode=mode, dropout_p=dropout_p)
    encoder.load_state_dict(torch.load(file_location+model_name+"_encoder.pt", map_location=device))
    encoder.eval()

    decoder = Decoder(hidden_size, train_out_nwords, max_length, layers=layers, mode=mode, dropout_p=dropout_p, attention=attention)
    decoder.load_state_dict(torch.load(file_location+model_name+"_decoder.pt", map_location=device))
    decoder.eval()

    return encoder, decoder