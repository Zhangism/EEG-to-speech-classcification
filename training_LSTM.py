# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:43:05 2020

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
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda")
all_data = np.load(save_path+'class_data.npy',allow_pickle=True)
label_class = [np.array([0,0,0,0,1]),np.array([1,0,0,1,0]),np.array([1,0,1,1,0]),np.array([0,0,0,1,0]),np.array([1,1,1,0,0]),np.array([1,1,0,0,0]),np.array([1,0,0,1,0]),np.array([1,0,1,0,0]),np.array([1,0,1,0,0]),np.array([1,1,0,0,0]),np.array([1,1,0,0,0])]
labels = []
task_list = ['Vowel only vs consonant','non-nasal vs nasal', 'non-bilabial vs bilabial ','non-iy vs iy ','non-uw vs uw']
for i in range(0,5):
    labels.append(np.load(save_path+'class'+str(i)+'.npy',allow_pickle=True))

#分为training，testing

#input大小 句子数量，帧数，特征数，需要将原始数据的后两维拉直
#network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm1 = nn.LSTM(62, 128,batch_first=True)
        self.lstm2 = nn.LSTM(128, 256,batch_first=True)
        self.classifier1 = nn.Linear(256, 1024)
        self.dp1 = nn.Dropout(p=0.5)
        self.classifier2 = nn.Linear(1024, 512)
        self.dp2 = nn.Dropout(p=0.5)
        self.classifier3 = nn.Linear(512, 2)
    def forward(self, x):
        out, (h_n, c_n) = self.lstm1(x)
        out, (h_n, c_n) = self.lstm2(out)
        x = out[:, -1, :]
        #print(x.shape)
        x = F.relu(self.dp1(self.classifier1(x)))
        x = F.relu(self.dp2(self.classifier2(x)))
        x = self.classifier3(x)
        return x

    def get_output(self, x):
        out, (h_n, c_n) = self.lstm1(x)
        out, (h_n, c_n) = self.lstm2(out)
        x = out[:, -1, :]
        #print(x.shape)
        x = F.relu(self.dp1(self.classifier1(x)))
        x = F.relu(self.dp2(self.classifier2(x)))
        return x  
    
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda")
net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.3)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)


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

for task in range(0,5):
    training_data_np = all_data[0:int(0.8*len(all_data))]
    training_label_np = labels[task][0:int(0.8*len(all_data))]
    training_data_tensor = torch.squeeze(torch.from_numpy(training_data_np).float())
    training_labels_tensor = torch.squeeze(torch.from_numpy(training_label_np.astype(np.float)).long())    
    training_set = TensorDataset(training_data_tensor,training_labels_tensor)
    train_loader = DataLoader(dataset = training_set, batch_size = 32, shuffle = True, num_workers = 0)
    
    testing_data_np = all_data[int(0.8*len(all_data)):]
    testing_label_np = labels[task][int(0.8*len(all_data)):]
    testing_data_tensor = torch.squeeze(torch.from_numpy(testing_data_np).float())
    testing_labels_tensor = torch.squeeze(torch.from_numpy(testing_label_np.astype(np.float)).long())    
    testing_set = TensorDataset(testing_data_tensor,testing_labels_tensor)
    test_loader = DataLoader(dataset = testing_set, batch_size = 32, shuffle = True, num_workers = 0)
    
    train_acc_np = np.zeros(50)
    test_acc_np = np.zeros(50)
    for epoch in range (0,50):
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
    torch.save(net, save_path+task_list[task]+'LSTM.pkl')