import os
import numpy as np

class DataLoader():
    def __init__(self, filepath):
        cwd = os.getcwd()
        self.basepath = cwd+"/"+filepath
        try:
            os.stat(self.basepath+"/add_prim_split")
            os.stat(self.basepath+"/few_shot_split")
            os.stat(self.basepath+"/filler_split")
            os.stat(self.basepath+"/length_split")
            os.stat(self.basepath+"/simple_split")
            os.stat(self.basepath+"/template_split")
        except Exception as e:
            raise Exception("Path "+filepath+" doesnt seem to contain the required folders.")

    def load_1a(self):
        train = []
        test = []

        with open(self.basepath+"/simple_split/tasks_train_simple.txt", "r") as f:
            for line in f:
                train.append(line_splitter(line))

        with open(self.basepath+"/simple_split/tasks_test_simple.txt", "r") as f:
            for line in f:
                test.append(line_splitter(line))

        return (np.asarray(train), np.asarray(test))

    def load_1b(self):
        percentile_dict = {}
        splits = ["1", "2", "4", "8", "16", "32", "64"]

        for percentile in splits:
            train = []
            test = []
            
            with open(self.basepath+"/simple_split/size_variations/tasks_train_simple_p{}.txt".format(percentile), "r") as f:
                for line in f:
                    train.append(line_splitter(line))

            with open(self.basepath+"/simple_split/size_variations/tasks_test_simple_p{}.txt".format(percentile), "r") as f:
                for line in f:
                    test.append(line_splitter(line))

            percentile_dict[percentile] = (np.asarray(train), np.asarray(test))
            
        return percentile_dict

def line_splitter(sentence):
    sent_list = sentence.split("OUT: ")
    sent_list[0] = sent_list[0].strip("IN: ")
    sent_list[1] = sent_list[1].strip("\n")

    return sent_list

dl = DataLoader("SCAN")

# examples:
# 1a :
#   train, test = dl.load_1a()
#   train[0][0] first train sentence, "IN"
#   train[0][1] first train sentence, "OUT"
# 1b :
#   dict = dl.load_1b()
#   train, test = dict["1"] extract the 1 percentile sentences out, split into train and test
#   train[0][0] first train sentence, "OUT"
#   train[0][1] first train sentence, "OUT"
#
# all returns are numpy arrays


import ipdb; ipdb.set_trace()