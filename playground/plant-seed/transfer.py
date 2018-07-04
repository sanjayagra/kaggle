import torch
import torchvision
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import itertools
from functools import partial

get_ipython().magic('matplotlib inline')

print('cuda:', torch.cuda.is_available())

transforms = []
transforms += [torchvision.transforms.Resize(224)]
transforms += [torchvision.transforms.RandomCrop(200)]
transforms += [torchvision.transforms.RandomHorizontalFlip()]
transforms += [torchvision.transforms.RandomVerticalFlip()]
transforms += [torchvision.transforms.RandomRotation(degrees=20)]
transforms += [torchvision.transforms.ToTensor()]
transforms += [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
transforms = torchvision.transforms.Compose(transforms)
trainset = torchvision.datasets.ImageFolder(root='../../data/plant-seed/data/train/', transform=transforms)

transforms = []
transforms += [torchvision.transforms.Resize(224)]
transforms += [torchvision.transforms.CenterCrop(200)]
transforms += [torchvision.transforms.ToTensor()]
transforms += [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
transforms = torchvision.transforms.Compose(transforms)
evalset = torchvision.datasets.ImageFolder(root='../../data/plant-seed/data/train/', transform=transforms)

transforms = []
transforms += [torchvision.transforms.Resize(224)]
transforms += [torchvision.transforms.CenterCrop(200)]
transforms += [torchvision.transforms.ToTensor()]
transforms += [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
transforms = torchvision.transforms.Compose(transforms)
validset = torchvision.datasets.ImageFolder(root='../../data/plant-seed/data/valid/', transform=transforms)

model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 12)
model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.)

def score(model, dataloader):
    
    calc_loss = 0.
    calc_correct = 0.
    calc_count = 0.
    
    for data in dataloader: 
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        
        calc_loss += loss.data[0]
        calc_correct += torch.sum(preds == labels.data)
        calc_count += outputs.data.shape[0]     
        
    return round(calc_loss/calc_count, 4), round(calc_correct/calc_count, 4)

def train(trainloader, evalloader, validloader, epochs):
    
    for epoch in range(epochs):
        calc_loss = 0.
        calc_correct = 0.
        calc_count = 0.
        
        for data in trainloader: 
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            calc_loss += loss.data[0]
            calc_correct += torch.sum(preds == labels.data)
            calc_count += outputs.data.shape[0]
        print('----------------')
        print('epoch:', epoch)
        print('run loss:', (round(calc_loss/calc_count, 4), round(calc_correct/calc_count, 4)))
        print('train loss:', score(model, evalloader))
        print('valid loss:', score(model, validloader))
        print('----------------')
            
    return model

def adjust_lr(optimizer, value):
    for param_group in optimizer.param_groups:
        param_group['lr'] = value
    return optimizer

train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
eval_loader  = torch.utils.data.DataLoader(evalset, batch_size=64, shuffle=False, num_workers=4)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=False, num_workers=4)

model = train(train_loader, eval_loader, valid_loader, 5)

optimzer = adjust_lr(optimizer, 0.00005)
model = train(train_loader, eval_loader, valid_loader, 5)

optimzer = adjust_lr(optimizer, 0.00001)
model = train(train_loader, eval_loader, valid_loader, 5)