import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os.path as osp


def get_activate_function(func_name):
    if func_name == 'leaky':
        func = F.leaky_relu
    elif func_name == 'elu':
        func = F.elu
    elif func_name == 'relu':
        func = F.relu
    elif func_name == 'sigmoid':
        func = F.sigmoid
    else:
        ValueError('Undetected function name {}'.format(func_name))
    return func


class fcnet(nn.Module):
    
    def __init__(self, batch_size, in_dim, hidden_dim, out_dim):
        self.batch_size = batch_size

        # Network layer definition
        self.bn0 = nn.BatchNorm1d(in_dim)
        self.dropout0 = nn.Dropout(0.2)
        
        self.fc1 = fully_block(in_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.activate_func1 = get_activate_function('elu')

        self.fc2 = fully_block(hidden_dim, out_dim)
        self.softmax = nn.Softmax()


    def forward(self, x):
        out = x
        out = self.bn0(out)
        out = self.dropout0(out)
        
        out = self.fc1(out)
        out = self.activate_func1(out)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.softmax(out)
        return out


class fully_block(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(fully_block, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)


    def forward(self, x):
        return self.fc(x)
    

def save_checkpoint(N, optim, score, checkpoint_dir, filename):
    state = {'Net': N,
            'optim': optim,
            'score': score}
    torch.save(state, osp.join(checkpoint_dir, filename))


def load_checkpoint(checkpoint_dir, filename):
    checkpoint = torch.load(osp.join(checkpoint_dir, filename))
    return checkpoint