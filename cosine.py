from embeddings import *
from data_loader import *
from training import *
from evaluate import *
from enc_and_dec import *
import scipy.spatial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dl = DataLoader("SCAN")
MAX_LENGTH = 100
file_location = "models/"

# BEST OVERALL MODEL
hidden_units=200
model='LSTM'

train_data, test_data = dl.load_1a()

train_in = Input("train_input")
train_out = Output("train_output")

test_in = Input("test_input")
test_out = Output("test_output")


for datapoint in train_data:
    train_in.addSentence(datapoint[0])
    train_out.addSentence(datapoint[1])

for datapoint in test_data:
    test_in.addSentence(datapoint[0])
    test_out.addSentence(datapoint[1])

encoder, decoder = load_models(train_in.n_words, train_out.n_words, hidden_units, 2, model, 0.5, False, file_location, "1st_LSTM_best_overall") #LSTM_no_att_drop_rerun1
hit, hit_idx, miss, miss_idx, hidden_state_vectors = evaluateItersAlt(train_data, encoder, decoder, train_in, train_out, vectors=True)

_, run_vec = evaluateAlt(encoder, decoder, "run", train_in, train_out, 1)

run_vectors = [(x[0], 1-scipy.spatial.distance.cosine(run_vec[0], x[1])) for x in hidden_state_vectors if x[0] != "jump"]

dtype = [('command', 'S100'), ('cosine', float)]
run_vectors = np.array(run_vectors, dtype=dtype)

run_vectors_sorted = np.sort(run_vectors, order="cosine")

print(run_vectors_sorted[len(run_vectors_sorted)-6:-1]) #in reverse order

# GRU, TASK SPECIFIC BEST MODEL
### JUMP
hidden_units=100
model='GRU'

data_dict = dl.load_3()
train_data, test_data = data_dict["jump"]

train_in = Input("train_input")
train_out = Output("train_output")

test_in = Input("test_input")
test_out = Output("test_output")

for datapoint in train_data:
    train_in.addSentence(datapoint[0])
    train_out.addSentence(datapoint[1])

for datapoint in test_data:
    test_in.addSentence(datapoint[0])
    test_out.addSentence(datapoint[1])

encoder, decoder = load_models(train_in.n_words, train_out.n_words, hidden_units, 1, model, 0.1, True, file_location, "3rd_GRU_100_dim_1_layer_att_0.1_drop5")
hit, hit_idx, miss, miss_idx, hidden_state_vectors = evaluateItersAlt(train_data, encoder, decoder, train_in, train_out, vectors=True)

_, jump_vec = evaluateAlt(encoder, decoder, "jump", train_in, train_out, vectors=True)

jump_vectors = []
jump_vectors = [(x[0], 1-scipy.spatial.distance.cosine(jump_vec[0], x[1])) for x in hidden_state_vectors if x[0] != "jump"]

dtype = [('command', 'S100'), ('cosine', float)]
jump_vectors = np.array(jump_vectors, dtype=dtype)

jump_vectors_sorted = np.sort(jump_vectors, order="cosine")

#print(jump_vectors_sorted[len(jump_vectors_sorted)-6:-1]) #in reverse order

#######################
### JUMP TWICE
_, jumpt_vec = evaluateAlt(encoder, decoder, "jump twice", train_in, train_out, vectors=True)

jumpt_vectors = [(x[0], 1-scipy.spatial.distance.cosine(jumpt_vec[0], x[1])) for x in hidden_state_vectors if x[0] != "jump"]

dtype = [('command', 'S100'), ('cosine', float)]
jumpt_vectors = np.array(jumpt_vectors, dtype=dtype)

jumpt_vectors_sorted = np.sort(jumpt_vectors, order="cosine")
import ipdb; ipdb.set_trace()
print(jumpt_vectors_sorted[len(jumpt_vectors_sorted)-6:-1]) #in reverse order