from data_loader import *
from embeddings import *

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, train_in, train_out, max_length=100):
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
			if encoder.mode == "LSTM":
				encoder_output, (encoder_hidden1, encoder_hidden2) = encoder(input_tensor[ei],(encoder_hidden1, encoder_hidden2))
			else:
				encoder_output, encoder_hidden1 = encoder(input_tensor[ei], encoder_hidden1)
			encoder_outputs[ei] += encoder_output[0, 0]

		decoder_input = torch.tensor([[SOS_token]], device=device)

		decoder_hidden1 = encoder_hidden1
		decoder_hidden2 = encoder_hidden2

		decoded_words = []
		
		for di in range(max_length):
			if decoder.mode == "LSTM":
				decoder_output, (decoder_hidden1, decoder_hidden2) = decoder(decoder_input, (decoder_hidden1, decoder_hidden2), encoder_outputs)
			else:
				decoder_output, decoder_hidden1 = decoder(decoder_input, decoder_hidden1, encoder_outputs)
			
			topv, topi = decoder_output.data.topk(1) 
			
			if topi.item() == EOS_token:
				break
			else:
				decoded_words.append(train_out.index2word[topi.item()])
		
		decoder_input = topi.squeeze().detach()

		return decoded_words

def evaluateIters(test_data, encoder, decoder, train_in, train_out):
	hit = 0
	miss = 0
	iters = 0
	hit_idx = []
	miss_idx = []

	for idx, test_point in enumerate(test_data):
		pred_list = evaluate(encoder, decoder, test_point[0], train_in, train_out)
		pred = " ".join(pred_list)

		if pred == test_point[1]:
			hit += 1
			hit_idx.append(idx)
		else:
			miss += 1
			miss_idx.append(idx)
		iters += 1

		if iters % 100 == 0:
			print(iters)
			print(hit)

	return hit, hit_idx, miss, miss_idx

def evaluate_and_save(test_data, model_name, save_file, encoder, decoder, train_in, train_out):
	print(encoder.hidden_size)
	hit, hit_idx, miss, miss_idx = evaluateIters(test_data, encoder, decoder, train_in, train_out)
	acc = 1-miss/len(test_data)

	with open("models/"+save_file, 'a') as f:
		f.write("Model name: " + model_name + "\n")
		f.write("Hits: " + str(hit) + "\n")
		f.write("Miss: " + str(miss) + "\n")
		f.write("Accuracy: " + str(acc) + "\n")

##################### ALTERNATIVE EVALUATIONS #####################

def evaluateAlt(encoder, decoder, sentence, train_in, train_out, vectors=False, actual_length=0, max_length=100):
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
			if encoder.mode == "LSTM":
				encoder_output, (encoder_hidden1, encoder_hidden2) = encoder(input_tensor[ei],(encoder_hidden1, encoder_hidden2))
			else:
				encoder_output, encoder_hidden1 = encoder(input_tensor[ei], encoder_hidden1)
			
			encoder_outputs[ei] += encoder_output[0, 0]

		decoder_input = torch.tensor([[SOS_token]], device=device)

		hidden_state_vector = encoder_hidden1
		decoder_hidden1 = encoder_hidden1
		decoder_hidden2 = encoder_hidden2

		decoded_words = []
		
		for di in range(max_length):
			if decoder.mode == "LSTM":
				decoder_output, (decoder_hidden1, decoder_hidden2) = decoder(decoder_input, (decoder_hidden1, decoder_hidden2), encoder_outputs)
			else:
				decoder_output, decoder_hidden1 = decoder(decoder_input, decoder_hidden1, encoder_outputs)
		
			topv, topi = decoder_output.data.topk(2)
				
			if topi[0][0].item() == EOS_token:
				if actual_length != 0:
					if len(decoded_words) == actual_length:
						break
					else:
						decoded_words.append(train_out.index2word[topi[0][1].item()])
				else:
					break
			elif topi[0][0].item() != EOS_token:
				decoded_words.append(train_out.index2word[topi[0][0].item()])
		
			topv, topi = decoder_output.data.topk(1)
			decoder_input = topi.squeeze().detach()

		if vectors:
			return decoded_words, hidden_state_vector
		else:
			return decoded_words

def evaluateItersAlt(test_data, encoder, decoder, train_in, train_out, oracle=False, vectors=False):
	hit = 0
	miss = 0
	iters = 0
	hit_idx = []
	miss_idx = []
	hidden_state_vectors = []

	for idx, test_point in enumerate(test_data):
		if oracle:
			pred_list = evaluateAlt(encoder, decoder, test_point[0], train_in, train_out, actual_length=len(test_point[1].split()))
		elif vectors:
			pred_list, hidden_state_vector = evaluateAlt(encoder, decoder, test_point[0], train_in, train_out, vectors=vectors)
			hidden_state_vectors.append((test_point[0],hidden_state_vector))
		else: 
			pred_list = evaluateAlt(encoder, decoder, test_point[0], train_in, train_out)
		
		pred = " ".join(pred_list)
		if pred == test_point[1]:
			hit += 1
			hit_idx.append(idx)
		else:
			miss += 1
			miss_idx.append(idx)
		iters += 1

		if iters % 500 == 0:
			print(iters)
			#print(hit)
	if vectors:
		return hit, hit_idx, miss, miss_idx, hidden_state_vectors
	else:
	  	return hit, hit_idx, miss, miss_idx

def evaluate_and_save(test_data, model_name, save_file, encoder, decoder, train_in, train_out):
	print(encoder.hidden_size)
	hit, hit_idx, miss, miss_idx = evaluateIters(test_data, encoder, decoder, train_in, train_out)
	acc = 1-miss/len(test_data)

	with open("models/"+save_file, 'a') as f:
		f.write("Model name: " + model_name + "\n")
		f.write("Hits: " + str(hit) + "\n")
		f.write("Miss: " + str(miss) + "\n")
		f.write("Accuracy: " + str(acc) + "\n")
