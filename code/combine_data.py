import numpy as np
import os.path as osp
import json
import os
import argparse


def create_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Folder contains raw features')
    parser.add_argument('combine_path', help='Output combined result ath')
    parser.add_argument('data_info_path', help='Information file that contains ground-truth')
    return parser


def load_data(data_path, data_list, in_dim=512, num_cls=103):
    num_data = len(data_list)
    x = np.empty((num_data, in_dim), dtype=np.float)
    y = np.empty(num_data, dtype=np.uint8)
    for i, data in enumerate(data_list):
        id, category = int(data['id']), int(data['category'])
        feature_data_path = osp.join(data_path, str(id) + '.npy')
        print(feature_data_path)
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


# def parse_json_info(json_fpath):
#     data = json.load(open(json_fpath, 'r'))
#     if not type(data) is dict:
#         data = json.loads(data)
#     return data


def create_folder(fold_path):
    if not osp.exists(fold_path):
        os.makedirs(fold_path)


def load_info_file(data_info_path, filename):
    data_info_filepath = osp.join(data_info_path, filename)
    data = []
    classes = []
    content = [line.rstrip() for line in open(data_info_filepath, 'r').readlines()][3:]
    cur_pt = 0
    while cur_pt < len(content):
        label, cls_id, num_items = content[cur_pt].split()
        classes.append(label)
        cls_id, num_items = int(cls_id), int(num_items)	
        for i in range(num_items):
            cur_pt += 1
            id = int(content[cur_pt])
            item = {'id' : id, 'category' : cls_id}
            data.append(item)
        cur_pt += 3
    return data, classes


if __name__ == '__main__':
    # data_path = osp.join(os.getcwd(), '..', 'data', 'landmark', 'features_rn50', 'raws')
    # json_fpath = osp.join(os.getcwd(), '..', 'data', 'landmark', 'train_val2018.json')
    # combine_path = osp.join(os.getcwd(), '..', 'data', 'landmark', 'features_rn50', 'combined_data_raw')

    # data = parse_json_info(json_fpath)
    args = create_argparse().parse_args()
    data, classes = load_info_file(args.data_info_path, 'SceneIBR_Image_Training.cla')
    cls_filepath = osp.join(args.data_info_path, 'classes.txt')
    if not osp.exists(cls_filepath):
        with open(cls_filepath, 'w') as f:
            for i in range(len(classes)):
                f.write('{}\n'.format(classes[i]))
    
    out_dims = len(classes)
    in_dim = 512
    x, y = load_data(args.data_path, data, in_dim=512, num_cls=out_dims)

    create_folder(args.combine_path)
    np.save(osp.join(args.combine_path, 'x_attributes.npy'), x)
    np.save(osp.join(args.combine_path, 'y_attributes.npy'), y)
