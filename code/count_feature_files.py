import os
import os.path as osp
import argparse


def create_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Feature folder')
    parser.add_argument('--ext', help='File extension')
    return parser


def test_file_count(args):
    cnt = 0
    if args.ext != None:
        for root, dirs, files in os.walk(args.data_path):
            for f in files:
                file_ext = f.split('.')[-1]
                if file_ext == args.ext:
                    cnt += 1
    else:
        for root, dirs, files in os.walk(args.data_path):
            cnt += len(files)
    return cnt


if __name__ == '__main__':
    args = create_argparse().parse_args()
    cnt = test_file_count(args)
    print('Total files: {}'.format(cnt))
