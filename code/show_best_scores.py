from fcnet import *
import os
import os.path as osp
import torch
import sys


checkpoint_path = '/home/jokernvt96/2DSceneCode/data/checkpoint'
chckpoint_filename = 'best.weights'
#chckpoint_filename = sys.argv[1]
chckpoint_filepath = osp.join(checkpoint_path, chckpoint_filename)

checkpoint = torch.load(chckpoint_filepath)
print(checkpoint['score'])
