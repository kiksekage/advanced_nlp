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

train_data = train_data[np.random.choice(100000, 1, replace=True), :]

file_location = "models/"
hidden_units=200
model='LSTM'
model_name="test_model"

#adjust the number of models you want to train: 1 iter = 1 model
for i in range(1, 2):
    train_and_save(train_data[:100], train_in, train_out, model, 0.0, False, 2, model_name+str(i), file_location, hidden_units=hidden_units)
    encoder, decoder = load_models(train_in.n_words, train_out.n_words, hidden_units, 2, model, 0.0, False, file_location, model_name+str(i))
    evaluate_and_save(test_data[:100], model_name+str(i), model_name+str(i)+".txt", encoder, decoder, train_in, train_out)
    print("################### ITERATION " +str(i) +" DONE ###################")