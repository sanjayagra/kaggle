import torch
import torchvision
import time
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.autograd import Variable
from torch.optim import lr_scheduler, Adam
from functools import reduce

torch.manual_seed(108)
torch.cuda.is_available()

data_transform = []
data_transform.append(transforms.RandomHorizontalFlip())
data_transform.append(transforms.ToTensor())
data_transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
data_transform = transforms.Compose(data_transform)

train_directory = datasets.ImageFolder(root='../data/train/', transform=data_transform)
train_generator = torch.utils.data.DataLoader(train_directory, batch_size=16, shuffle=True, num_workers=4)

valid_directory = datasets.ImageFolder(root='../data/valid/', transform=data_transform)
valid_generator = torch.utils.data.DataLoader(valid_directory, batch_size=16, shuffle=True, num_workers=4)

model1 = models.resnet152(pretrained=True)
model1 = torch.nn.Sequential(*list(model1.children())[:-1])

model2 = models.resnet101(pretrained=True)
model2 = torch.nn.Sequential(*list(model2.children())[:-1])

for param in model1.parameters():
    param.requires_grad = False

for param in model2.parameters():
    param.requires_grad = False

class dense_model(torch.nn.Module):
    
    def __init__(self, model1, model2):
        super(dense_model, self).__init__()
        self.model1 = model1
        self.model2 = model2
        layers = []
        layers += [torch.nn.Linear(2048+2048, 1000)]
        layers += [torch.nn.Dropout(p=0.5)]
        layers += [torch.nn.Linear(1000, 120)]
        self.fc = torch.nn.Sequential(*layers) 
        return None
    
    def forward(self,x):
        score1 = self.model1(x)
        score2 = self.model2(x)
        score1 = score1.view(score1.size(0), -1)
        score2 = score2.view(score2.size(0), -1)
        score = torch.cat((score1,score2),1)
        score = score.view(score.size(0), -1)
        score = self.fc(score)
        return score

model = dense_model(model1, model2)

def execute(model, optimizer, num_epochs):
    
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train(True)
        running_loss = 0.
        running_corrects = 0.
        for data in train_generator:
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / 9199
        epoch_accuracy = running_corrects / 9199
        print('epoch {} completed..'.format(epoch))
        print('train loss: {:.4f},  train accuracy: {:.4f}'.format(epoch_loss, epoch_accuracy))
        
        model.train(False)
        running_loss = 0.
        running_corrects = 0.
        for data in valid_generator:
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / 1023
        epoch_accuracy = running_corrects / 1023
        print('valid loss: {:.4f}, valid accuracy: {:.4f}'.format(epoch_loss, epoch_accuracy))
            
    return model

optimizer = Adam(model.fc.parameters(), lr=0.0001)
model = execute(model, optimizer, 3)
torch.save(model.state_dict(), '../data/model/resnet_stacked.pth')

data_transform = []
data_transform.append(transforms.ToTensor())
data_transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
data_transform = transforms.Compose(data_transform)

score_directory = datasets.ImageFolder(root='../data/test/', transform=data_transform)
score_generator = torch.utils.data.DataLoader(score_directory, batch_size=16, shuffle=False, num_workers=4)

outputs = []
for data in score_generator:
    inputs, labels = data
    inputs = Variable(inputs.cuda())
    scores = model(inputs)
    scores = torch.nn.functional.softmax(scores)
    outputs.append(np.array(scores.data.cpu().numpy()))
outputs = reduce(lambda x,y : np.vstack((x,y)), outputs)
print('scored file:', outputs.shape)
pd.DataFrame(outputs).to_csv('../data/score/resnet_stacked.csv', index=False, header=False)