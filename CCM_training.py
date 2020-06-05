# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:20:24 2020

@author: zhang
"""

import os
import numpy as np
import torch

from torch.utils.data import DataLoader,TensorDataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt

save_path = 'Q:\\大学\\毕业设计\\代码\\'
FEATURE = 'CCM'
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda")
task = 0
epoches=200
'''
1: Vowel only (0) vs consonant (1)
2: non-nasal (0) vs nasal (1)
3: non-bilabial (0) vs bilabial (1)
4: non-iy (0) vs iy (1)
5: non-uw (0) vs uw (1)
'''
#load data
all_data = np.load(save_path+'class_data.npy',allow_pickle=True)
label_class = [np.array([0,0,0,0,1]),np.array([1,0,0,1,0]),np.array([1,0,1,1,0]),np.array([0,0,0,1,0]),np.array([1,1,1,0,0]),np.array([1,1,0,0,0]),np.array([1,0,0,1,0]),np.array([1,0,1,0,0]),np.array([1,0,1,0,0]),np.array([1,1,0,0,0]),np.array([1,1,0,0,0])]
labels = []
task_list = ['Vowel only vs consonant','non-nasal vs nasal', 'non-bilabial vs bilabial ','non-iy vs iy ','non-uw vs uw']
for i in range(0,5):
    labels.append(np.load(save_path+'class'+str(i)+'.npy',allow_pickle=True))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 25 * 25, 64)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 128)
        self.dp2 = nn.Dropout(p=0.5)        
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.dp1(F.relu(self.fc1(x)))
        x = self.dp2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def get_output(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.dp1(F.relu(self.fc1(x)))
        x = self.dp2(F.relu(self.fc2(x)))
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
def train(epoch):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i,data in enumerate(train_loader):
        #inputs,labels = data
        inputs,labels = data[0].to(device),data[1].to(device)
        inputs,labels = Variable(inputs),Variable(labels)
        optimizer.zero_grad()        
        outputs = net(inputs)
        #print(torch.squeeze(outputs).shape)
        #print(labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    #print(epoch,'training:Loss: %.3f | Acc: %.3f%% (%d/%d)'% (running_loss/(i+1), 100.*correct/total, correct, total))
    train_acc_np[epoch] = 100.*correct/total
    
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs,labels = data[0].to(device),data[1].to(device)
            inputs,labels = Variable(inputs),Variable(labels)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            #print(outputs.max(1),labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #eprint(sum(predicted))
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    #print(epoch, 'testing:Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(i+1), 100.*correct/total, correct, total))
    test_acc_np[epoch] = 100.*correct/total
   
#converting into tensor
for task in range(0,5):
    training_data_np = all_data[0:int(0.8*len(all_data))]
    training_label_np = labels[task][0:int(0.8*len(all_data))]
    training_data_tensor = torch.from_numpy(training_data_np).float()
    training_labels_tensor = torch.from_numpy(training_label_np).long()
    training_set = TensorDataset(training_data_tensor,training_labels_tensor)
    train_loader = DataLoader(dataset = training_set, batch_size = 32, shuffle = True, num_workers = 0)
    
    testing_data_np = all_data[int(0.8*len(all_data)):]
    testing_label_np = labels[task][int(0.8*len(all_data)):]
    testing_data_tensor = torch.from_numpy(testing_data_np).float()
    testing_labels_tensor = torch.from_numpy(testing_label_np).long()
    testing_set = TensorDataset(testing_data_tensor,testing_labels_tensor)
    test_loader = DataLoader(dataset = testing_set, batch_size = 32, shuffle = True, num_workers = 0)

    net = CNN()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=0.01)
    train_acc_np = np.zeros(epoches)
    test_acc_np = np.zeros(epoches)
    for epoch in range (0,epoches):
        train(epoch)
        test(epoch)
    print(task_list[task])
    print(max(test_acc_np))
    plt.figure(figsize=(9, 2))
    plt.plot(test_acc_np,label = 'test',dashes=[6, 1])
    plt.plot(train_acc_np,label = 'train')
    plt.xlabel('epoch')
    plt.ylabel('acc(%)')
    plt.title(task_list[task]+' binary')
    plt.grid()
    plt.savefig("Q:\\大学\\毕业设计\\图片\\"+task_list[task]+'2binary CCM'+'.jpg',dpi=300)   
    plt.show()

    torch.save(net, save_path+task_list[task]+'CNN.pkl')
##循环输出所有结果
#保存网络再调用，不然内存不够
#model = torch.load(save_path+task_list[task]+'CNN.pkl')
#net.get_output(training_data_tensor.to(device)).shape