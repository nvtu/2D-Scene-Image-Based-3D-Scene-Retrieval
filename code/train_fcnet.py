from data_loader import *
from fcnet import *
from tqdm import tqdm
import os.path as osp
import os
import torch.nn as nn


if __name__ == '__main__':
    combined_path = osp.join(os.getcwd(), '..', 'data', 'landmark', 'features', 'combined_data')
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

    N = fcnet(batch_size, in_dim, hidden_dim, out_dim)
    data = DataLoader(batch_size, combined_path)
    N = N.to(device)
    cost_func = nn.BCELoss()
    optim = torch.optim.Adam(N.parameters(), lr=learning_rate)

    best_score = 0

    for i in tqdm(range(total_steps)):
        x, y = data.next_batch()
        x = torch.Tensor(x).to(device)
        y = torch. Tensor(y).to(device)
        out = N(x)
        loss = cost_func(out, y)
        loss.backward()
        optim.step()
        N.zero_grad()

        if i % print_step == 0:
            cnt = 0
            matched = []
            while cnt < data.cnt_test:
                x_test, y_test = data.next_test()
                cnt += len(y_test)
                result = N(x_test)
                _, predicts = result.sort(1)
                predicts = predicts[:,-3:]
                ground_truth = y_test.argmax(1)

                m = [True if ground_truth[j] in predicts[j] else False for j in range(batch_size)]
                matched += m
            score = matched.sum() * 1.0 / data.cnt_test
            print("Iters: {}  - Loss: {} - Accuracy: {}".format(i, loss, score))

            if score > best_score:
                best_score = score
                save_checkpoint(N, optim, score, checkpoint_dir, "best_weights.chohuu")