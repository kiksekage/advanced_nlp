from GRU import *
from embeddings import *
from data_loader import *
from training import *
from evaluate import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dl = DataLoader("SCAN")
train_data, test_data = dl.load_1a()

train_in = Lang("train_input")
train_out = Lang("train_output")

test_in = Lang("test_input")
test_out = Lang("test_output")

for datapoint in train_data:
        train_in.addSentence(datapoint[0])
        train_out.addSentence(datapoint[1])

encoder_layers=2
decoder_layers=2
encoder_dropout=0.5
decoder_dropout=0.5

encoder = GRUEncoder(train_in.n_words, 200, encoder_layers, encoder_dropout)
decoder = GRUDecoder(200, train_out.n_words, decoder_layers, decoder_dropout)

losses = trainIters(encoder, decoder, train_data, train_in, train_out, trials=10000)

miss = evaluateIters(test_data, encoder, decoder, train_in, train_out)

import ipdb; ipdb.set_trace()
