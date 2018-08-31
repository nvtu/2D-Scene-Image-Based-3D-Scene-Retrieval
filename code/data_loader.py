import numpy as np
import os.path
import json
import os.path as osp
import os


class DataLoader():
    def __init__(self, batch_size, data_path, public_test=False):
        self.batch_size = batch_size
        self.data_path = data_path

        if public_test == False:
            self.x = np.load(osp.join(self.data_path, 'x_attributes.npy'))
            self.y = np.load(osp.join(self.data_path, 'y_attributes.npy'))

            self.cnt_total = len(self.y)
            self.cnt_val = int(self.cnt_total / 5.0) # Keep the ratio between train/test data split to be 80/20
            self.cnt_train = self.cnt_total - self.cnt_val
        else:
            self.x = np.load(osp.join(self.data_path, 'x_attributes.npy'))
            self.cnt_total = self.x.shape[0]
            self.cnt_test = self.cnt_total

        self.permutation = np.random.permutation(self.cnt_total)

        # Some variables determine the current data pointer index
        self.pt_train_index = self.cnt_val
        self.pt_test_index = 0
        self.pt_val_index = 0


    def next_batch(self):
        start_index = self.pt_train_index
        end_index = start_index + self.batch_size
        if end_index > self.cnt_total: # May miss data at the end !!!
            np.random.shuffle(self.permutation[self.cnt_val:self.cnt_total])
            start_index = self.cnt_val
            end_index = start_index + self.batch_size
        index = self.permutation[start_index:end_index]
        self.pt_train_index = end_index
        return self.x[index], self.y[index]


    def next_val(self):
        start_index = self.pt_val_index
        end_index = start_index + self.batch_size
        if end_index > self.cnt_val:
            index = self.permutation[start_index:self.cnt_val]
            self.pt_val_index = 0
            return self.x[index], self.y[index]
        self.pt_val_index = end_index
        index = self.permutation[start_index:end_index]
        return self.x[index], self.y[index]
    
    
    def next_test(self):
        start_index = self.pt_test_index
        end_index = start_index + self.batch_size
        if end_index > self.cnt_test:
            index = self.permutation[start_index:self.cnt_test]
            self.pt_test_index = 0
            return self.x[index]
        self.pt_test_index = end_index
        index = self.permutation[start_index:end_index]
        return self.x[index]
