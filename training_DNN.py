# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:19:40 2020

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
FEATURE = 'Linear'
input_len = 12
NET = 'leaky_NoDropout'
subject = 0
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda")
final_accuracy = []
epochs = 50
class Net_drop_leaky(nn.Module):
    def __init__(self):
        super(Net_drop_leaky, self).__init__()
        self.fc1 = nn.Linear(17*62*input_len, 5000)
        self.dp1 = nn.Dropout(p=0.05)
        self.fc2 = nn.Linear(5000, 5000)
        self.dp2 = nn.Dropout(p=0.05)
        self.fc3 = nn.Linear(5000, 5000)
        self.dp3 = nn.Dropout(p=0.05)
        self.fc4 = nn.Linear(5000, 3000)
        self.dp4 = nn.Dropout(p=0.05)
        self.fc5 = nn.Linear(3000, 11)
        
    def forward(self, x):
        x = x.view(-1,17*62*input_len)
        x = F.leaky_relu(self.dp1(self.fc1(x)), negative_slope=0.001, inplace=False)  
        x = F.leaky_relu(self.dp2(self.fc2(x)), negative_slope=0.001, inplace=False)  
        x = F.leaky_relu(self.dp3(self.fc3(x)), negative_slope=0.001, inplace=False)  
        x = F.leaky_relu(self.dp4(self.fc4(x)), negative_slope=0.001, inplace=False)  
        x = self.fc5(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(17*62*input_len, 5000)
        self.fc2 = nn.Linear(5000, 5000)
        self.fc3 = nn.Linear(5000, 5000)
        self.fc4 = nn.Linear(5000, 3000)
        self.fc5 = nn.Linear(3000, 11)
    def forward(self, x):
        x = x.view(-1,17*62*input_len)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class Net_leaky(nn.Module):
    def __init__(self):
        super(Net_leaky, self).__init__()
        self.fc1 = nn.Linear(17*62*input_len, 5000)
        self.fc2 = nn.Linear(5000, 5000)
        self.fc3 = nn.Linear(5000, 5000)
        self.fc4 = nn.Linear(5000, 3000)
        self.fc5 = nn.Linear(3000, 11)
        
    def forward(self, x):
        x = x.view(-1,17*62*input_len)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.001, inplace=False)  
        x = F.leaky_relu(self.fc2(x), negative_slope=0.001, inplace=False)  
        x = F.leaky_relu(self.fc3(x), negative_slope=0.001, inplace=False)  
        x = F.leaky_relu(self.fc4(x), negative_slope=0.001, inplace=False)  
        x = self.fc5(x)
        return x
class Net_drop(nn.Module):
    def __init__(self):
        super(Net_drop, self).__init__()
        self.fc1 = nn.Linear(17*62*input_len, 5000)
        self.dp1 = nn.Dropout(p=0.05)
        self.fc2 = nn.Linear(5000, 5000)
        self.dp2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(5000, 5000)
        self.dp3 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(5000, 3000)
        self.dp4 = nn.Dropout(p=0.05)
        self.fc5 = nn.Linear(3000, 11)
         
    def forward(self, x):
        x = x.view(-1,17*62*input_len)
        x = F.relu(self.dp1(self.fc1(x)))
        x = F.relu(self.dp2(self.fc2(x)))
        x = F.relu(self.dp3(self.fc3(x)))
        x = F.relu(self.dp4(self.fc4(x)))
        x = self.fc5(x)
        return x
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
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    print(epoch,'training:Loss: %.3f | Acc: %.3f%% (%d/%d)'% (running_loss/(i+1), 100.*correct/total, correct, total))
    train_acc_np[epoch] = 100.*correct/total
    train_loss_np[epoch] = running_loss/(i+1)

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
            loss = criterion(outputs, labels)
            #print(outputs.max(1),labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(epoch, 'testing:Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(i+1), 100.*correct/total, correct, total))
    test_acc_np[epoch] = 100.*correct/total
    test_loss_np[epoch] = test_loss/(i+1)
    return correct/total
all_data = np.load(save_path+FEATURE+'data'+'.npy', allow_pickle=True)    
for subject in range(0,11):
    data = np.zeros((all_data[subject].shape[0],all_data[subject][0][0].shape[0],all_data[subject][0][0].shape[1],all_data[subject][0][0].shape[2]))#165*17*310
    for i in range(0,all_data[subject].shape[0]):#165
        for j in range (0,17):
            #data[i][j] = np.ravel(all_data[subject][i][0][j])
            data[i][j] = all_data[subject][i][0][j]
    
    #feature_num = all_data[0][0].shape[3]
    #分为training，testing
    training = data[0:int(0.8*len(data))]
    training_data_np = np.array(training)
    training_label_np = all_data[subject][:,1][0:int(0.8*len(data))]
    training_data_tensor = torch.from_numpy(training_data_np).float()
    training_labels_tensor = torch.squeeze(torch.from_numpy(training_label_np.astype(np.float)).long())    
    training_set = TensorDataset(training_data_tensor,training_labels_tensor)
    train_loader = DataLoader(dataset = training_set, batch_size = 64, shuffle = True, num_workers = 0)
    
    testing = data[int(0.8*len(data)):]
    testing_data_np = np.array(testing)
    testing_label_np = all_data[subject][:,1][int(0.8*len(data)):]
    testing_data_tensor = torch.from_numpy(testing_data_np).float()
    testing_labels_tensor = torch.squeeze(torch.from_numpy(testing_label_np.astype(np.float)).long())    
    testing_set = TensorDataset(testing_data_tensor,testing_labels_tensor)
    test_loader = DataLoader(dataset = testing_set, batch_size = 64, shuffle = True, num_workers = 0)

    if NET == 'Normal':
        net = Net()
    elif NET == 'leaky_NoDropout':
        net = Net_leaky()
    elif NET == 'Leaky_Dropout':
        net = Net_drop_leaky()
    elif NET == 'NoLeaky_Dropout' :
        net = Net_drop()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.3)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00004)
    train_acc_np = np.zeros(epochs)
    test_acc_np = np.zeros(epochs)
    train_loss_np = np.zeros(epochs)
    test_loss_np = np.zeros(epochs)
    for epoch in range (0,epochs):
        train(epoch)
        test(epoch)
    plt.figure(figsize=(9, 2))
    plt.plot(test_loss_np,label = 'test',dashes=[6, 1])
    plt.plot(train_loss_np,label = 'train')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(FEATURE+' '+NET+' loss '+'Subject %d'%(subject+1))
    plt.grid()
    plt.savefig("Q:\\大学\\毕业设计\\图片\\网络结构比较\\"+"\\"+FEATURE+' '+NET+' loss'+'Subject %d'%(subject+1)+'.jpg',dpi=300)    
    plt.show()
    
    plt.figure(figsize=(9, 2))
    plt.plot(test_acc_np,label = 'test',dashes=[6, 1])
    plt.plot(train_acc_np,label = 'train')
    plt.xlabel('epoch')
    plt.ylabel('Acc(%)')
    plt.title(FEATURE+' '+NET+' acc '+'Subject %d'%(subject+1))
    plt.grid()
    plt.savefig("Q:\\大学\\毕业设计\\图片\\网络结构比较\\"+"\\"+FEATURE+' '+NET+' acc'+'Subject %d'%(subject+1)+'.jpg',dpi=300)    
    plt.show()
    final_accuracy.append(max(test_acc_np))
np.save(save_path+NET+FEATURE+'ACC'+'.npy',final_accuracy)
plt.figure(figsize=(9, 3))
plt.bar(range(len(final_accuracy)),final_accuracy)
plt.show()