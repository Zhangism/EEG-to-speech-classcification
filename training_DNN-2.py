
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:19:40 2020

@author: zhang
"""
'''
1: Vowel only (0) vs consonant (1)
2: non-nasal (0) vs nasal (1)
3: non-bilabial (0) vs bilabial (1)
4: non-iy (0) vs iy (1)
5: non-uw (0) vs uw (1)
'''
task_list = ['Vowel only vs consonant','non-nasal vs nasal', 'non-bilabial vs bilabial ','non-iy vs iy ','non-uw vs uw']

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
import random
save_path = 'Q:\\大学\\毕业设计\\代码\\'
FEATURE = 'MFCC'
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda")
final_accuracy = []
task = 0
class Net_drop_leaky(nn.Module):
    def __init__(self):
        super(Net_drop_leaky, self).__init__()
        self.fc1 = nn.Linear(17*62*13, 5000)
        self.bn1 = nn.BatchNorm1d(5000, momentum=0.1)
        self.fc2 = nn.Linear(5000, 5000)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(5000, 5000)
        self.fc4 = nn.Linear(5000, 3000)
        self.bn2 = nn.BatchNorm1d(3000, momentum=0.1)
        self.fc5 = nn.Linear(3000, 2)
        
    def forward(self, x):
        x = x.view(-1,17*62*13)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.001, inplace=False)  
        x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.001, inplace=False) 
        x = self.dp1(x)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.001, inplace=False)  
        x = F.leaky_relu(self.fc4(x), negative_slope=0.001, inplace=False)  
        x = self.bn2(x)
        x = self.fc5(x)
        return x
    '''def output(self, x):
        x = x.view(-1,17*62*13)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.002, inplace=False) 
        x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.002, inplace=False)  
        x = F.leaky_relu(self.fc3(x), negative_slope=0.002, inplace=False)  
        x = self.bn2(x)
        x = self.fc4(x)
        return x'''

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
        outputs = net(torch.squeeze(inputs,1))
        loss = criterion(torch.squeeze(outputs,1), labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        out = net(torch.squeeze(inputs,1))
        #_, predicted = out.max(1)
        predicted = torch.max(outputs, 1)[1]
        total += labels.size(0)
        #predicted_np = predicted.data.numpy()
        #target = labels.data.numpy()
        #print(outputs)
        #print(out)
        #print(torch.squeeze(outputs,1))
        #print(sum(labels))
        #print(sum(torch.squeeze(outputs,1)))
        #print(predicted)
        '''for i in range(0,len(predicted)):
            #print(predicted[i])
            if predicted[i]>0:
                predicted[i]=1
            else:
                predicted[i]=0'''
        correct += sum(predicted.eq(labels))
        #correct += predicted.eq(labels).sum().item()
    print(epoch,'training:Loss: %.3f | Acc: %.3f%% (%d/%d)'% (running_loss/(i+1), 100.*correct/total, correct, total))
    train_acc_np[epoch] = 100.*correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs,labels = data[0].to(device),data[1].to(device)
            inputs,labels = Variable(inputs),Variable(labels)
            outputs = net(torch.squeeze(inputs, 1))
            loss = criterion(torch.squeeze(outputs,1), labels)
            #print(outputs.max(1),labels)
            test_loss += loss.item()
            #_, predicted = outputs.max(1)
            total += labels.size(0)
            '''for i in range(0,len(predicted)):
                if predicted[i]>0:
                    predicted[i]=1
                else:
                    predicted[i]=0'''
            predicted = torch.max(outputs, 1)[1]
            correct += predicted.eq(labels).sum().item()
    print(epoch, 'testing:Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(i+1), 100.*correct/total, correct, total))
    test_acc_np[epoch] = 100.*correct/total
    return correct/total

all_data = np.load(save_path+'MFCC'+'data'+'.npy', allow_pickle=True)
data = []
for subject in range(0,11):
    for item in range(0,len(all_data[subject])):
        data.append(all_data[subject][item])
random.shuffle(data)
label_class = [np.array([0,0,0,0,1]),np.array([1,0,0,1,0]),np.array([1,0,1,1,0]),np.array([0,0,0,1,0]),np.array([1,1,1,0,0]),np.array([1,1,0,0,0]),np.array([1,0,0,1,0]),np.array([1,0,1,0,0]),np.array([1,0,1,0,0]),np.array([1,1,0,0,0]),np.array([1,1,0,0,0])]
labels_binary = np.zeros((len(data),5))
data_np = np.zeros([1484,17,62,13])
for i in range(0,len(data)):
    labels_binary[i] = label_class[int(data[i][1])]
    data_np[i] = data[i][0]
labels_binary_np = np.array(labels_binary)
for task in range(0,5):
    
    label = labels_binary_np[:,task]
    training_data_np = data_np[0:int(0.8*len(data_np))]
    training_label_np = label[0:int(0.8*len(label))]
    training_data_tensor = torch.from_numpy(training_data_np).float()
    training_labels_tensor = torch.squeeze(torch.from_numpy(training_label_np.astype(np.float)).long())
    training_set = TensorDataset(training_data_tensor,training_labels_tensor)
    train_loader = DataLoader(dataset = training_set, batch_size = 64, shuffle = True, num_workers = 0)
    
    testing_data_np = data_np[int(0.8*len(data_np)):]
    testing_label_np = label[int(0.8*len(label)):]
    testing_data_tensor = torch.from_numpy(testing_data_np).float()
    testing_labels_tensor = torch.squeeze(torch.from_numpy(testing_label_np.astype(np.float)).long()) 
    testing_set = TensorDataset(testing_data_tensor,testing_labels_tensor)
    test_loader = DataLoader(dataset = testing_set, batch_size = 64, shuffle = True, num_workers = 0)
    
    net = Net_drop_leaky()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.3)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00005)
    train_acc_np = np.zeros(100)
    test_acc_np = np.zeros(100)
    for epoch in range (0,100):
        train(epoch)
        acc = round(test(epoch),3)
        if epoch == 99:
            final_accuracy.append(acc)
    fig, ax = plt.subplots()
    ax.plot(test_acc_np,label = 'test',dashes=[6, 1])
    ax.plot(train_acc_np,label = 'train')
    ax.set(xlabel='epoch', ylabel='acc(%)')
    plt.title(task_list[task]+' binary')
    plt.text(99,acc,acc,ha='center', va='bottom', fontsize=13)
    ax.grid()
    ax.legend()
    plt.savefig("Q:\\大学\\毕业设计\\图片\\"+task_list[task]+'2binary '+'.jpg',dpi=300)    
    plt.show()