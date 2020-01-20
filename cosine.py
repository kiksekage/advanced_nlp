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
'''
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

vectors = []
for x in hidden_state_vectors:
    vectors.append((x[0], 1-scipy.spatial.distance.cosine(run_vec[1], x[1][1].cpu())))

dtype = [('command', 'S100'), ('cosine', float)]
vectors = np.array(vectors, dtype=dtype)

vectors_sorted = np.sort(vectors, order="cosine")

print(vectors_sorted[len(vectors_sorted)-6:-1]) #in reverse order
'''
# GRU, TASK SPECIFIC BEST MODEL
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

_, run_vec = evaluateAlt(encoder, decoder, "run", train_in, train_out, 1)

vectors = []
for x in hidden_state_vectors:
    vectors.append((x[0], 1-scipy.spatial.distance.cosine(run_vec[1], x[1][1].cpu())))

dtype = [('command', 'S100'), ('cosine', float)]
vectors = np.array(vectors, dtype=dtype)

vectors_sorted = np.sort(vectors, order="cosine")

print(vectors_sorted[len(vectors_sorted)-6:-1]) #in reverse order