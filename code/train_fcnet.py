from data_loader import *
from fcnet import *
from tqdm import tqdm
import os.path as osp
import os
import torch.nn as nn


if __name__ == '__main__':
    data_path = osp.join(os.getcwd(), '..', 'data', 'landmark', 'features', 'attributes')
    json_path = osp.join(os.getcwd(), '..', 'data', 'landmark', 'train_val2018.json')
    checkpoint_dir = osp.join(os.getcwd(), '..', 'data', 'landmark', 'checkpoint')


    total_steps = 256000
    print_step = 10
    save_step = 1000
    batch_size = 1000
    learning_rate = 1e-4
    in_dim = 102
    hidden_dim = 2000
    out_dim = 103

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    print(device)

    N = fcnet(batch_size, in_dim, hidden_dim, out_dim)
    data = DataLoader(batch_size, data_path, json_path)
    N = N.to(device)
    cost_func = nn.BCELoss()
    optim = torch.optim.Adam(N.parameters(), lr=learning_rate)


    begin_step = 0
    best_score = 0

    for i in tqdm(range(begin_step, total_steps)):
        x, y = data.next_batch()
        x = torch.Tensor(x).to(device)
        y = torch. Tensor(y).to(device)
        out = N(x)
        loss = cost_func(out, y)
        loss.backward()
        optim.step()
        N.zero_grad()

        if i % print_step == 0:
            _, predicts = out.sort(1)
            predicts = predicts[:,-3:]
            ground_truth = y.argmax(1)
            match = np.array([True if ground_truth[j] in predicts[j] else False for j in range(batch_size)]).sum()
            print("Iters: {}  - Loss: {} - Accuracy: {}".format(i, loss, match))
        if i % save_step == 0:
            save_checkpoint(N, optim, match, checkpoint_dir, '{}_{}'.format(i, match))
