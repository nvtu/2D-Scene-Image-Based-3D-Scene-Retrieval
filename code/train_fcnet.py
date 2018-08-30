from data_loader import *
from fcnet import *
from tqdm import tqdm
import os.path as osp
import os


if __name__ == '__main__':
    data_path = osp.join(os.getcwd(), '..', 'data', 'landmark', 'features', 'attributes')
    json_path = osp.join(os.getcwd(), '..', 'data', 'landmark', 'train_val2018.json')

    total_steps = 256000
    print_step = 1000
    save_step = 1000
    batch_size = 128
    learning_rate = 1e-4
    in_dim = 102
    hidden_dim = 200
    out_dim = 103

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    N = fcnet(batch_size, in_dim, hidden_dim, out_dim)
    data = DataLoader(batch_size, data_path, json_path)
    N = N.to(device)
    cost_func = nn.BCELoss(True)
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
            print("Iters: {}  - Loss: {}".format(i, loss))