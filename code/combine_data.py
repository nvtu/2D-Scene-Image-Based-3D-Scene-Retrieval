import numpy as np
import os.path as osp
import json
import os


def load_data(data_path, data_list, in_dim=102, num_cls=103):
    num_data = len(data_list)
    x = np.empty((num_data, in_dim), dtype=np.float)
    y = np.empty(num_data, dtype=np.uint8)
    for i, data in enumerate(data_list):
        id, category = int(data['id']), int(data['category'])
        feature_data_path = osp.join(data_path, str(category), str(id) + '.npy')
        if osp.isfile(feature_data_path):
            _data = np.load(feature_data_path)
        x[i] = _data
        y[i] = category
    y = to_one_hot(y, num_cls)
    return x, y


def to_one_hot(y, num_cls):
    num_data = len(y)
    onehot = np.zeros((num_data, num_cls), dtype=np.uint8)   
    for i in range(num_data):
        onehot[i][y[i]] = 1
    return onehot


def parse_json_info(json_fpath):
    data = json.load(open(json_fpath, 'r'))
    if not type(data) is dict:
        data = json.loads(data)
    return data


if __name__ == '__main__':
    data_path = osp.join(os.getcwd(), '..', 'data', 'landmark', 'features', 'attributes')
    json_fpath = osp.join(os.getcwd(), '..', 'data', 'landmark', 'train_val2018.json')
    combine_path = osp.join(os.getcwd(), '..', 'data', 'landmark', 'combined_data')

    data = parse_json_info(json_fpath)
    x, y = load_data(data_path, data)

    np.save(osp.join(combine_path, 'x_attributes.npy'), x)
    np.save(osp.join(combine_path, 'y_attributes.npy'), y)