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

# Oracle experiment
hidden_units=50
model='GRU'
model_name="oracle_gru"

train_data, test_data = dl.load_2()

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

encoder, decoder = load_models(train_in.n_words, train_out.n_words, hidden_units, 2, model, 0.5, True, file_location, model_name)
hit, hit_idx, miss, miss_idx = evaluateItersAlt(test_data, encoder, decoder, train_in, train_out, oracle=True)

print(1-miss/test_data.shape[0])