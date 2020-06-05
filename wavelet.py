# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:55:32 2020

@author: zhang
"""
'''
四阶Daubechies 小波变换
d1-d5和a5
输出165x[17x62x5]=165x5270
'''
import pywt
from pywt import wavedec
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
#from sklearn.metrics import confusion_matrix
#import seaborn as sns
#import matplotlib.pyplot as plt
path = '\\new\\thinkeing.mat'
dataMat=loadmat('A:\\MM\\MM08\\new\\thinking.mat')
array = dataMat['thinking_mats']
db4 = pywt.Wavelet('db4')



epochs = array[0]
epoch_wd = np.zeros([array.shape[1],17,62,500])
count = 0
for epoch in epochs:#epoch.shape = 62x4900
    #print(epoch.shape)
    for i in range(0,17):
        epoch_wd[count][i]=epoch[...,0+i*250:500+i*250]
    count=count+1
X = epoch_wd
X_wave = np.zeros([array.shape[1],17,62,5])
for i in range(0,X.shape[0]):
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            A5, D5, D4, D3, D2, D1 = np.square(wavedec(X[i][j][k], db4,level = 5))
            a5 = np.sum(A5)
            d5 = np.sum(D5)
            d4 = np.sum(D4)
            d3 = np.sum(D3)
            d2 = np.sum(D2)
            ET = np.sum([a5, d5, d4, d3, d2])
            X_wave[i][j][k][0] = a5/ET
            X_wave[i][j][k][1] = d5/ET
            X_wave[i][j][k][2] = d4/ET
            X_wave[i][j][k][3] = d3/ET
            X_wave[i][j][k][4] = d2/ET
            
#load data
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda")
Y = np.empty([array.shape[1],1])
count = 0
dict = {'/uw/': 0,'/tiy/': 1, '/piy/': 2,'/iy/': 3,'/m/': 4,'/n/': 5,'/diy/': 6,'pat': 7,'pot': 8,'gnaw': 9,'knew': 10}

with open("A:\\MM\\MM08\\kinect_data\\labels.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        Y[count] = dict[line]
        count = count + 1
X_t = torch.from_numpy(X_wave[0:int(0.8*len(X_wave))]).float()
Y_t = torch.squeeze(torch.from_numpy(Y[0:int(0.8*len(Y))]).long())
X_test = torch.from_numpy(X_wave[int(0.8*len(X_wave)):]).float()
Y_test = torch.squeeze(torch.from_numpy(Y[int(0.8*len(Y)):]).long())
dataset_t = TensorDataset(X_t,Y_t)
train_loader = DataLoader(dataset = dataset_t, batch_size = 4, shuffle = True, num_workers = 0)

dataset_test = TensorDataset(X_test,Y_test)
test_loader = DataLoader(dataset = dataset_t, batch_size = 1, shuffle = True, num_workers = 0)
#network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(17*62*5, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, 300)
        self.fc4 = nn.Linear(300, 11)
        
    def forward(self, x):
        x = x.view(-1,17*62*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.6)
for epoch in range (200):
    running_loss = 0.0
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
        if i % 5 == 4:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0
#PATH = './net.pth'
#torch.save(net.state_dict(), PATH)
#Y_p = np.empty([165,1])
correct = 0
total = 0
#count = 0
with torch.no_grad():
    for data in test_loader:
        #datas, labels = data
        inputs,labels = data[0].to(device),data[1].to(device)
        inputs,labels = Variable(inputs),Variable(labels)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        print(outputs)
        print(predicted.cpu().numpy()[0])
        #Y_p[count]=predicted.cpu().numpy()[0]
        #count = count + 1
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network: %d %%' % (100 * correct / total))
'''
label_list = ['/uw/','/tiy/', '/piy/','/iy/','/m/','/n/','/diy/','pat','pot','gnaw','knew']
y_true = Y.tolist()
y_pred = Y_p.tolist()
CM=confusion_matrix(y_true, y_pred, labels=label_list, sample_weight=None, normalize=None)
f,ax=plt.subplots()
sns.heatmap(CM,annot=True,ax=ax)'''
