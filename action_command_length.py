from embeddings import *
from data_loader import *
from training import *
from evaluate import *
from enc_and_dec import *

from collections import defaultdict
# count up the command and action lengths

dl = DataLoader("SCAN")
MAX_LENGTH = 100
file_location = "models/"

train_data, test_data = dl.load_1a()

train_in = Input("train_input")
train_out = Output("train_output")

for datapoint in train_data:
    train_in.addSentence(datapoint[0])
    train_out.addSentence(datapoint[1])

test_in = Input("test_input")
test_out = Output("test_output")

for datapoint in test_data:
    test_in.addSentence(datapoint[0])
    test_out.addSentence(datapoint[1])

model='LSTM'
hidden_units=200
model_name = "1st_LSTM_best_overall"

action_length_dict = defaultdict(int)
command_length_dict = defaultdict(int)

for point in test_data:
    command_length=len(point[0].split())
    action_length=len(point[1].split())
    action_length_dict[action_length] +=1
    command_length_dict[command_length] +=1


total_accuracy_command_length_dict = defaultdict(list)
total_accuracy_action_length_dict = defaultdict(list)

encoder, decoder = load_models(train_in.n_words, train_out.n_words, hidden_units, 2, model, 0.5, False, file_location, model_name)
hit, hit_idx, miss, miss_idx = evaluateItersAlt(test_data, encoder, decoder, train_in, train_out)

accuracy_command_length_dict = defaultdict(int)
accuracy_action_length_dict = defaultdict(int)

for idx in miss_idx:
    command_length=len(test_data[idx][0].split())
    action_length=len(test_data[idx][1].split())
    accuracy_action_length_dict[action_length] += 1
    accuracy_command_length_dict[command_length] += 1

for key in accuracy_action_length_dict:
    action_length_acc = 1-accuracy_action_length_dict[key]/action_length_dict[key]
    total_accuracy_action_length_dict[key].append(action_length_acc)

for key in accuracy_command_length_dict:
    command_length_acc = 1-accuracy_command_length_dict[key]/command_length_dict[key]
    total_accuracy_command_length_dict[key].append(command_length_acc)

import ipdb; ipdb.set_trace()