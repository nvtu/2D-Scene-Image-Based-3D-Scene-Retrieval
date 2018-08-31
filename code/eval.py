import torch
from data_loader import *
from fcnet import *
from tqdm import tqdm
from combine_data import *
import os
import os.path as osp
import argparse


def create_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_filepath', help='Path of checkpoint file')
    parser.add_argument('result_filename', help='Where to save the csv result')
    return parser


def combine_features(data_path, combine_path, in_dim=102):
    combined_filename = 'x_attributes.npy'
    combine_path = osp.join(combine_path, combined_filename)
    if osp.isfile(combine_path):
        print("All features are already combined")
        return

    features_filenames = sorted(os.listdir(data_path))
    num_data = len(features_filenames)
    x = np.empty((num_data, in_dim), dtype=np.float)
    print("Combining features...")
    for i, f in enumerate(features_filenames):
        data_filepath = osp.join(data_path, f)
        data = np.load(data_filepath)
        x[i] = data
    np.save(combine_path, x)


def load_network(checkpoint_dir, filename):
    checkpoint = load_checkpoint(checkpoint_dir, filename)
    N = checkpoint['Net']
    return N


def eval(N, data_path):
    data = DataLoader(N.batch_size, data_path, public_test=True)
    cnt = 0
    results = []
    top = 3
    while cnt < data.cnt_total:
        x = data.next_test()
        cnt += x.shape[0]
        out = N(x)
        _, predicts = out.sort(1)
        predicts = predicts[:,-top:]
        results += predicts
    return results


def dump_csv(results, data_path, result_filepath):
    list_imgs = sorted(os.listdir(data_path))
    num_imgs = len(list_imgs)
    with open(result_filepath, 'w') as f:
        f.write('id,predicted\n')
        for i in range(num_imgs):
            out = results[i]
            img_name = results[i].split('.')[0]
            f.write('{},'.format(img_name))
            for j in range(len(out)):
                if j == len(out) - 1: 
                    f.write('{}\n'.format(out[j]))
                else: 
                    f.write('{} '.format(out[j]))


if __name__ == '__main__':
    args = create_argparse().parse_args()

    data_path = osp.join(os.getcwd(), '..', 'data', 'landmark', 'features_public', 'attributes')
    combine_path = osp.join(os.getcwd(), '..', 'data', 'landmark', 'features_public', 'combined_data')
    combine_features(data_path, combine_path)
    
    checkpoint_dir = osp.join(os.getcwd(), '..', 'data', 'landmark', 'checkpoint')
    checkpoint_filename = args.checkpoint_filepath
    N = load_network(checkpoint_dir, checkpoint_filename)

    results = eval(N, combine_path)
    result_filepath = osp.join(os.getcwd(), '..', 'data', 'landmark', 'results')
    result_filename = args.result_filename
    dump_csv(results, data_path, osp.join(result_filepath, result_filename))