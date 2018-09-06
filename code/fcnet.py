import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os.path as osp

def get_activate_function(func_name):
    if func_name == 'leaky':
#        func = F.leaky_relu
        func = nn.LeakyReLU()
    elif func_name == 'elu':
#        func = F.elu
        func = nn.ELU()
    elif func_name == 'sigmoid':
        func = nn.Sigmoid()
    else:
        ValueError('Undetected function name {}'.format(func_name))
    return func


class fcnet(nn.Module):
    
    def __init__(self, batch_size, in_dim, hidden_dim, out_dim):
        super(fcnet, self).__init__()
        self.batch_size = batch_size

        # Network layer definition
        self.dropout = nn.Dropout(0.2)
        self.activate_func = get_activate_function('elu')
        self.bn0 = nn.BatchNorm1d(in_dim)
        
        self.fc_block = fully_block([in_dim, hidden_dim, hidden_dim, out_dim], self.activate_func)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()


    def forward(self, x):
        out = x
        out = self.bn0(out)
#        out = self.dropout(out)

        out = self.fc_block(out)
#        out = self.activate_func(out)
        out = self.softmax(out)
#        out = self.sigmoid(out)
        return out


class fully_block(nn.Module):

    def __init__(self, dims, activate_func):
        super(fully_block, self).__init__()
        last = None
        fcs = []
#        gg = dims[-1] # bad code
        for i, cur_dim in enumerate(dims):
            if (last != None):
                fcs.append(nn.Linear(last, cur_dim))
                fcs.append(activate_func)
                fcs.append(nn.BatchNorm1d(cur_dim))
#                if i  == len(dims) - 1:
#                    fcs.append(nn.Dropout(0.2))
#                    #if gg!= cur_dim:
                #    fcs.append(activate_func)
                #    fcs.append(nn.BatchNorm1d(cur_dim))
                #    # Try adding dropout layer
            last = cur_dim

        self.fc = nn.Sequential(*fcs)

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
