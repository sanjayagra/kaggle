import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

class csv_data(Dataset):
    
    def __init__(self, filename):
        self.dataframe = pd.read_csv(filename).astype('float32')
        return None
    
    def __getitem__(self, index):
        X, y = self.dataframe.iloc[index,:-1], self.dataframe.iloc[index,-1]
        y = np.atleast_1d(np.array(y, dtype=np.int64))
        X, y = torch.from_numpy(np.array(X, dtype=np.float64)), torch.LongTensor(y).squeeze()
        return X.float(), y.squeeze()
    
    def __len__(self):
        return self.dataframe.shape[0]

train_resnet = csv_data('../../data/plant-seed/data/features/train_resnet_50.csv')
valid_resnet = csv_data('../../data/plant-seed/data/features/valid_resnet_50.csv')

train_resnet = DataLoader(train_resnet, batch_size=12, shuffle=True, num_workers=1)
valid_resnet = DataLoader(valid_resnet, batch_size=64, shuffle=False, num_workers=1)

class resnet(torch.nn.Module):

    def __init__(self):
        super(resnet, self).__init__()
        self.dense1 = torch.nn.Linear(2048, 500)
        self.dropout = torch.nn.Dropout(p=0.33)
        self.batchnorm = torch.nn.BatchNorm1d(num_features=500)
        self.dense2 = torch.nn.Linear(500, 12)
        return None

    def forward(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.dense2(x)
        return x
    
model = resnet()
model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.)

def score(model, dataloader):
    calc_loss = 0.
    calc_correct = 0.    
    for data in dataloader: 
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels.squeeze())
        calc_loss += loss.data[0]
        calc_correct += torch.sum(preds == labels.squeeze().data) 
    count = len(dataloader.dataset)
    return round(calc_loss/count, 4), round(calc_correct/count, 4)

def train(trainloader, evalloader, validloader, epochs):
    for epoch in range(epochs):
        calc_loss = 0.
        calc_correct = 0.
        for data in trainloader: 
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            calc_loss += loss.data[0]
            calc_correct += torch.sum(preds == labels.squeeze().data)
        count = len(trainloader.dataset)
        print('----------------')
        print('epoch:', epoch)
        print('run loss:', (round(calc_loss/count, 4), round(calc_correct/count, 4)))
        print('train loss:', score(model, evalloader))
        print('valid loss:', score(model, validloader))
        print('----------------')     
    return model

def adjust_lr(optimizer, value):
    for param_group in optimizer.param_groups:
        param_group['lr'] = value
    return optimizer

optimizer = adjust_lr(optimizer, 0.0001)
model = train(train_resnet, train_resnet, valid_resnet, 1)

optimizer = adjust_lr(optimizer, 0.0005)
model = train(train_resnet, train_resnet, valid_resnet, 1)

optimizer = adjust_lr(optimizer, 0.0001)
model = train(train_resnet, train_resnet, valid_resnet, 2)

optimizer = adjust_lr(optimizer, 0.00005)
model = train(train_resnet, train_resnet, valid_resnet, 2)

optimizer = adjust_lr(optimizer, 0.00001)
model = train(train_resnet, train_resnet, valid_resnet, 2)