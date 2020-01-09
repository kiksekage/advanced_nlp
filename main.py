from embeddings import *
from data_loader import *
from training import *
from evaluate import *
from enc_and_dec import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#USED IN COLAB
#dl = DataLoader("/content/drive/My Drive/Colab Notebooks/SCAN")

#USED ON OWN PC
dl = DataLoader("SCAN")

#MAX_LENGTH = max([len(x[0].split()) for x in train_data]) + 1
MAX_LENGTH = 100

#DATA LOADING AND LANGUAGE CREATION, DIFFERS BETWEEN EXERCISES
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

train_data = train_data[np.random.choice(train_data.shape[0], 1, replace=True), :]

plt.rcParams['figure.figsize'] = [10, 6]

file_location = "models/"
hidden_units=200
model='LSTM'

for i in range(1, 1):
    train_and_save(train_data, train_in, train_out, model, 0.0, False, 2, "LSTM_no_att_no_drop"+str(i), file_location, hidden_units=hidden_units)
    encoder, decoder = load_models(train_in.n_words, train_out.n_words, hidden_units, 2, model, 0.0, False, file_location, "LSTM_no_att_no_drop"+str(i))
    evaluate_and_save(train_data, "LSTM_no_att_no_drop"+str(i), "LSTM_no_att_no_drop"+str(i)+".txt", encoder, decoder, train_in, train_out, model)
    print("################### ITERATION " +str(i) +" DONE ###################")