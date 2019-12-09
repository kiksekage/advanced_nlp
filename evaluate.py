import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import *
from embeddings import *
from layers_attempt import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=100):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                #decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        return decoded_words

def evaluateIters(test_data, encoder, decoder, lang_in, lang_out):
    miss = 0
    iters = 0

    for test_point in test_data:
        pred = evaluate(encoder, decoder, test_point[0], lang_in, lang_out)
        pred = " ".join(pred)
        if pred != test_point[1]:
            miss += 1
        iters += 1

        if iters % 100 == 0:
            print(iters)
            print(miss)

    return miss