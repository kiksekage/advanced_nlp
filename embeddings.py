from data_loader import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SOS_token = 0
EOS_token = 1

class Input:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        #self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.index2word = {}
        #self.n_words = 2  # Count SOS and EOS
        self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class Output:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        #self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.index2word = {}
        #self.n_words = 2  # Count SOS and EOS
        self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


        
def get_embedding(word, lookup_dict, embeds):
    tensor = torch.tensor([lookup_dict[word]], dtype=torch.long)
    return embeds(tensor)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    output_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, output_tensor)