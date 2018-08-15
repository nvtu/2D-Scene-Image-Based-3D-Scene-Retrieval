import numpy as np
import json

class DataLoader():


    def __init__(self, batch_size, data_path, json_fpath):
        self.batch_size = batch_size
        self.data_path = data_path
        self.data_info = self.__parse_json_info(json_fpath)
        self.cnt_total = len(self.data_info)
        self.cnt_test = int(self.cnt_total / 5.0) # Keep the ratio between train/test data split to be 80/20
        self.cnt_train = self.cnt_total - self.cnt_test

        self.permutation = np.random.permutation(self.cnt_total)

        # Some variables determine the current data pointer index
        self.pt_train_index = self.cnt_test
        self.pt_test_index = 0

    
    def __parse_json_info(self, json_fpath):
        data = json.load(open(json_fpath, 'r'))
        if not type(data) is dict:
            data = json.loads(data)
        return data


    def next_batch(self):
        start_index = self.pt_train_index
        end_index = start_index + self.batch_size
        if end_index > self.cnt_total: # May miss data at the end !!!
            np.random.shuffle(self.permutation[self.cnt_test:self.cnt_total])
            start_index = self.cnt_test
            end_index = start_index + self.batch_size
        index = self.permutation[start_index:end_index]
        self.pt_train_index = end_index
        return self.data_info[index]
    

    def next_test(self):
        start_index = self.pt_test_index
        end_index = start_index + self.batch_size
        if end_index > self.cnt_test:
            start_index = 0
        end_index = start_index + self.batch_size
        self.pt_test_index = end_index
        index = self.permutation[start_index:end_index]
        return self.data_info[index]