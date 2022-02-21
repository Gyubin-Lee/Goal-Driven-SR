import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import collections
import time
import progressbar
from multiprocessing import Pool

# define train and test function
def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target, reduction = "mean")
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(output.item(), target.item())
            test_loss += F.mse_loss(output, target.reshape((target.size(dim=0), 1)), reduction='sum').item()

    test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

# define DNN
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        final = self.fc4(h3)

        return final

#TODO save result and analyze
def deque_mean(deque):
    sum = 0
    len = 0
    while deque:
        sum += deque.pop()
        len += 1
    return sum/len

# load data
NUM_SUBJECT = 63
df = pd.read_csv('SR_bias_fitting_result.csv', sep=',')
target = torch.from_numpy(df['CES_D'].values.astype(np.float32)).view(NUM_SUBJECT, 1)
features = torch.from_numpy(df.drop('CES_D', axis = 1).values.astype(np.float32))

def work_func(p):
    # result
    NUM_PERMUT = 10000
    result = np.empty(shape=(NUM_SUBJECT, 1) , dtype = object)
    for i in range(NUM_SUBJECT):
        result[i][0] = collections.deque()

    # set hyperparameters
    NUM_EPOCH = 10
    LR = 0.01
    seed = 15237
    log_interval = 4
    NUM_TEST = int(p * NUM_SUBJECT)
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    time.sleep(1)

    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # split data into train and test set
    bar = progressbar.ProgressBar(maxval = NUM_PERMUT).start()
    for prgs in range(NUM_PERMUT):
        bar.update(prgs)

        model = DNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr = LR)

        test_ind = np.random.choice(range(63), NUM_TEST, replace = False)
        train_ind = np.setdiff1d(np.arange(63), (test_ind))

        train_features = features[train_ind]
        train_target = target[train_ind]
        train_set = torch.utils.data.TensorDataset(train_features, train_target)
        test_features = features[test_ind]
        test_target = target[test_ind]
        test_set = torch.utils.data.TensorDataset(test_features, test_target)


        train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = 4)
        test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = 1)

        # train model and test the performance of the model
        for epoch in range(1, NUM_EPOCH + 1):
            train(log_interval, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)

        with torch.no_grad():
            output = model(test_features.to(device))
            ind = 0
            for sub in test_ind.tolist():
                result[sub][0].append(output[ind].item())
                ind += 1

    bar.finish()

    filename = './DNN_result/result0120_p' + str(p)
    np.save(filename, result)

    result_mean = np.zeros((63, 1), dtype = np.float32)
    for sub in range(63):
        result_mean[sub][0] = deque_mean(result[sub][0])

    print(result_mean)

TEST_RATE = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

for p in TEST_RATE:
    work_func(p)



    
