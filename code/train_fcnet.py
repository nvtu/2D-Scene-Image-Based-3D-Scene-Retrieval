from data_loader import *
from fcnet import *
from tqdm import tqdm
import os.path as osp
import os
import torch.nn as nn

def eval_test(N, top, step):
    cnt = 0
    matched = []
    while cnt < data.cnt_val:
        x_test, y_test = data.next_val()
        x_test = torch.Tensor(x_test).to(device)
        y_test = torch.Tensor(y_test).to(device)
        cnt += len(y_test)
        result = N(x_test)
        _, predicts = result.sort(1)
        predicts = predicts[:,-top:].cpu().numpy()
        ground_truth = y_test.argmax(1).cpu().numpy()

        m = [True if ground_truth[j] in predicts[j] else False for j in range(ground_truth.__len__())]
        matched += m
    score = np.array(matched).sum() * 1.0 / data.cnt_val
    return score


def calcTrainScore(N, top, step = -1):
    _, predicts = out.sort(1)
    predicts = predicts[:,-top:]
    ground_truth = y.argmax(1)
    m = [True if ground_truth[j] in predicts[j] else False for j in range(ground_truth.__len__())]
    score_train = np.array(m).sum() * 1.0 / len(ground_truth)
    return score_train

if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    combined_path = osp.join(os.getcwd(), '..', 'data', 'landmark', 'features_rn50', 'combined_data_raw')
    json_path = osp.join(os.getcwd(), '..', 'data', 'landmark', 'train_val2018.json')
    checkpoint_dir = osp.join(os.getcwd(), '..', 'data', 'landmark', 'checkpoint')


    total_steps = 256000
    print_step = 1000
    save_step = 1000
    batch_size = 1000
    learning_rate = 1e-4
    in_dim = 512
    hidden_dim = 1024
    out_dim = 30

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    N = fcnet(batch_size, in_dim, hidden_dim, out_dim)
    data = DataLoader(batch_size, combined_path)
    N = N.to(device)
#    cost_func = nn.MSELoss()
#    cost_func = nn.NLLLoss()
    cost_func = nn.BCELoss()
    optim = torch.optim.Adam(N.parameters(), lr=learning_rate)

    best_score = 0

    for i in tqdm(range(total_steps)):
        x, y = data.next_batch()
        x = torch.Tensor(x).to(device)
        y = torch. Tensor(y).to(device)
        tam = y.argmax(1)
        out = N(x)
        #loss = cost_func(out, tam)
        loss = cost_func(out, y)
        loss.backward()
        optim.step()
        N.zero_grad()

        if i % print_step == 0:
            score = eval_test(N, 1, i)           
            # Calculate training accuracy score to prevent overfit 

            score_train = calcTrainScore(N, 1, i)
            print("Iters: {}  - Loss: {} - Accuracy Train: {} - Accuracy Test: {}".format(i, loss, score_train, score))
            save_checkpoint(N, optim, score, checkpoint_dir, str(i))

            if score > best_score:
                best_score = score
                save_checkpoint(N, optim, score, checkpoint_dir, "best.weights")
