import os
import numpy as np
import random

class DataLoader():
    def __init__(self, filepath):
        cwd = os.getcwd()
        self.basepath = filepath
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
        train = self.file_loader("/simple_split/tasks_train_simple.txt")
        test = self.file_loader("/simple_split/tasks_test_simple.txt")

        return (np.asarray(train), np.asarray(test))

    def load_1b(self):
        percentile_dict = {}
        splits = ["1", "2", "4", "8", "16", "32", "64"]

        for percentile in splits:
            train = self.file_loader("/simple_split/size_variations/tasks_train_simple_p{}.txt".format(percentile))
            test = self.file_loader("/simple_split/size_variations/tasks_test_simple_p{}.txt".format(percentile))
            
            percentile_dict[percentile] = (np.asarray(train), np.asarray(test))
            
        return percentile_dict

    def load_2(self):
        train = self.file_loader("/length_split/tasks_train_length.txt")
        test = self.file_loader("/length_split/tasks_test_length.txt")

        return (np.asarray(train), np.asarray(test))

    def load_3(self):
        """
        loads the datasets for both parts of the experiment
        the first part where both primitives appear without compositional commands
        the second part where 'jump' primitive appears in
        compositional commands of varying lengths
        returns a dictionary of pairs all possible train/test sets
        """
        data_dict = {}
        nums = ["1", "2", "4", "8", "16", "32"]
        reps = ["1", "2", "3", "4", "5"]

        train = self.file_loader("/add_prim_split/tasks_train_addprim_jump.txt")
        test = self.file_loader("/add_prim_split/tasks_test_addprim_jump.txt")
        data_dict['jump'] = (np.asarray(train), np.asarray(test))

        train = self.file_loader("/add_prim_split/tasks_train_addprim_turn_left.txt")
        test = self.file_loader("/add_prim_split/tasks_test_addprim_turn_left.txt")
        data_dict['lturn'] = (np.asarray(train), np.asarray(test))
        
        for num in nums:
            for rep in reps:
                train = self.file_loader("/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num{}_rep{}.txt".format(num, rep))
                test = self.file_loader("/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num{}_rep{}.txt".format(num, rep))
                
                data_dict['jump_num{}_rep{}'.format(num, rep)] = (np.asarray(train), np.asarray(test))
            
        return data_dict

    def file_loader(self, path):
        sent_list = []
        with open(self.basepath+path, "r") as f:
                    for line in f:
                        sent_list.append(line_splitter(line))
        return sent_list

    
def line_splitter(sentence):
    sent_list = sentence.split("OUT: ")
    sent_list[0] = sent_list[0].strip("IN: ")
    sent_list[1] = sent_list[1].strip("\n")

    return sent_list

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
