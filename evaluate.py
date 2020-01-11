from data_loader import *
from embeddings import *

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, train_in, train_out, mode, max_length=100):
    with torch.no_grad():
        input_tensor = tensorFromSentence(train_in, sentence)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            encoder.cuda()
            decoder.cuda()

        input_length = input_tensor.size()[0]
        encoder_hidden1 = torch.zeros(encoder.layers, 1, encoder.hidden_size, device=device)
        encoder_hidden2 = torch.zeros(encoder.layers, 1, encoder.hidden_size, device=device)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
          if mode == "LSTM":
            encoder_output, (encoder_hidden1, encoder_hidden2) = encoder(input_tensor[ei],
                                                     (encoder_hidden1, encoder_hidden2))
            encoder_outputs[ei] += encoder_output[0, 0]
          else:
            encoder_output, encoder_hidden1 = encoder(input_tensor[ei], encoder_hidden1)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden1 = encoder_hidden1
        decoder_hidden2 = encoder_hidden2

        decoded_words = []
        for di in range(max_length):
          if mode == "LSTM":
            decoder_output, (decoder_hidden1, decoder_hidden2) = decoder(decoder_input, (decoder_hidden1, decoder_hidden2), encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(train_out.index2word[topi.item()])
          else:
            decoder_output, decoder_hidden1 = decoder(decoder_input, decoder_hidden1, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(train_out.index2word[topi.item()])


            decoder_input = topi.squeeze().detach()
        return decoded_words

def evaluateIters(test_data, encoder, decoder, train_in, train_out, mode):
    hit = 0
    miss = 0
    iters = 0
    hit_idx = []
    miss_idx = []

    for idx, test_point in enumerate(test_data):
        pred_list = evaluate(encoder, decoder, test_point[0], train_in, train_out, mode)
        pred = " ".join(pred_list)
        if pred == test_point[1]:
            hit += 1
            hit_idx.append(idx)
        else:
            miss += 1
            miss_idx.append(idx)
        iters += 1

        #if iters % 100 == 0:
        #    print(iters)
        #    print(hit)

    return hit, hit_idx, miss, miss_idx

def evaluate_and_save(test_data, model_name, save_file, encoder, decoder, train_in, train_out, mode):
    print(encoder.hidden_size)
    hit, hit_idx, miss, miss_idx = evaluateIters(test_data, encoder, decoder, train_in, train_out, mode)
    acc = 1-miss/len(test_data)

    with open(save_file, 'a') as f:
        f.write("Model name: " + model_name + "\n")
        f.write("Hits: " + str(hit) + "\n")
        f.write("Miss: " + str(miss) + "\n")
        f.write("Accuracy: " + str(acc) + "\n")
