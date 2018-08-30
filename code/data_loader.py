import numpy as np
import os.path
import json
import os.path as osp
import os


class DataLoader():
    def __init__(self, batch_size, data_path, json_fpath):
        self.batch_size = batch_size
        self.data_path = data_path
        self.data_info = np.array(self.__parse_json_info(json_fpath))
#        self.x, self.y = self.__load_data(self.data_info)
#        np.save(self.data_path + '/../chocamx', self.x)
#        np.save(self.data_path + '/../chocamy', self.y)
        self.x = np.load(self.data_path + '/../chocamx.npy')
        self.y = np.load(self.data_path + '/../chocamy.npy')


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

    
    def __to_one_hot(self, y, num_cls):
        num_data = len(y)
        onehot = np.zeros((num_data, num_cls), dtype=np.uint8)   
        for i in range(num_data):
            onehot[i][y[i]] = 1
        return onehot


    def __load_data(self, data_list, in_dim=102, num_cls=103):
        num_data = len(data_list)
        x = np.empty((num_data, in_dim), dtype=np.float)
        y = np.empty(num_data, dtype=np.uint8)
        for i, data in enumerate(data_list):
            id, category = int(data['id']), int(data['category'])
            feature_data_path = osp.join(self.data_path, str(category), str(id) + '.npy')
            if os.path.isfile(feature_data_path):
                _data = np.load(feature_data_path)
            x[i] = _data
            y[i] = category
        y = self.__to_one_hot(y, num_cls)
        return x, y


    def next_batch(self):
        start_index = self.pt_train_index
        end_index = start_index + self.batch_size
        if end_index > self.cnt_total: # May miss data at the end !!!
            np.random.shuffle(self.permutation[self.cnt_test:self.cnt_total])
            start_index = self.cnt_test
            end_index = start_index + self.batch_size
        index = self.permutation[start_index:end_index]
        self.pt_train_index = end_index
        return self.x[index], self.y[index]

    def next_test(self):
        start_index = self.pt_test_index
        end_index = start_index + self.batch_size
        if end_index > self.cnt_test:
            start_index = 0
        end_index = start_index + self.batch_size
        self.pt_test_index = end_index
        index = self.permutation[start_index:end_index]
        return self.x[index], self.y[index]
