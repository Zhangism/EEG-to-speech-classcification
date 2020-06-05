# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:43:24 2020

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
epoches = 100
'''
label_class = [np.array([0,0,0,0,1]),np.array([1,0,0,1,0]),np.array([1,0,1,1,0]),np.array([0,0,0,1,0]),np.array([1,1,1,0,0]),np.array([1,1,0,0,0]),np.array([1,0,0,1,0]),np.array([1,0,1,0,0]),np.array([1,0,1,0,0]),np.array([1,1,0,0,0]),np.array([1,1,0,0,0])]
labels = []
for i in range(0,5):
    labels.append(np.load(save_path+'class'+str(i)+'.npy',allow_pickle=True))'''
task_list = ['Vowel only vs consonant','non-nasal vs nasal', 'non-bilabial vs bilabial ','non-iy vs iy ','non-uw vs uw']



class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 1152),
            )
        
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode
    def get_output(self,x):
        return self.encoder(x)
    
def train(epoch):
    net_DAE.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i,data in enumerate(train_loader):
        inputs,labels = data[0].to(device),data[1].to(device)
        inputs,labels = Variable(inputs),Variable(labels)
        optimizer.zero_grad()        
        outputs = net_DAE(inputs)
        #print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        #_, predicted = outputs.max(1)
        #total += labels.size(0)
        #correct += predicted.eq(labels).sum().item()
    #print(epoch,'training:Loss: %.3f | Acc: %.3f%% (%d/%d)'% (running_loss/(i+1), 100.*correct/total, correct, total))
    train_loss_np[epoch] = running_loss/(i+1)

def test(epoch):
    net_DAE.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs,labels = data[0].to(device),data[1].to(device)
            inputs,labels = Variable(inputs),Variable(labels)
            outputs = net_DAE(inputs)
            loss = criterion(outputs, labels)
            #print(outputs.max(1),labels)
            test_loss += loss.item()
            #running_loss += loss.item()
            #_, predicted = outputs.max(1)
            #print(sum(predicted))
            #total += labels.size(0)
            #correct += predicted.eq(labels).sum().item()
    #print(epoch, 'testing:Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(i+1), 100.*correct/total, correct, total))
    test_loss_np[epoch] = test_loss/(i+1)
    


for task in range(0,5):
    #label_np = labels[task]
    #labels_tensor = torch.squeeze(torch.from_numpy(label_np.astype(np.float)).long())
    net_DAE = autoencoder()
    net_DAE.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net_DAE.parameters(), lr=0.001)
    all_data1 = np.load(save_path+task_list[task]+'CNN.npy')
    all_data2 = np.load(save_path+task_list[task]+'LSTM.npy')
    all_data = np.hstack((all_data1,all_data2))
    data_tensor = torch.from_numpy(all_data).float()    
    
    training_data_np = all_data[0:int(0.9*len(all_data))]
    training_label_np = all_data[0:int(0.9*len(all_data))]
    training_data_tensor = torch.from_numpy(training_data_np).float()
    training_labels_tensor = torch.from_numpy(training_label_np).float()
    training_set = TensorDataset(training_data_tensor,training_labels_tensor)
    train_loader = DataLoader(dataset = training_set, batch_size = 64, shuffle = True, num_workers = 0)
    
    testing_data_np = all_data[int(0.1*len(all_data)):]
    testing_label_np = all_data[int(0.1*len(all_data)):]
    testing_data_tensor = torch.from_numpy(testing_data_np).float()
    testing_labels_tensor = torch.from_numpy(testing_label_np).float()
    testing_set = TensorDataset(testing_data_tensor,testing_labels_tensor)
    test_loader = DataLoader(dataset = testing_set, batch_size = 64, shuffle = True, num_workers = 0)
    train_loss_np = np.zeros(epoches)
    test_loss_np = np.zeros(epoches)
    for epoch in range(0,epoches):
        train(epoch)
        test(epoch)
    print(task_list[task])
    print(max(test_loss_np))
    plt.figure(figsize=(9, 2))
    plt.plot(test_loss_np,label = 'test',dashes=[6, 1])
    plt.plot(train_loss_np,label = 'train')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(task_list[task]+' binary')
    plt.grid()
    plt.savefig("Q:\\大学\\毕业设计\\图片\\"+task_list[task]+'2binary CCM'+'.jpg',dpi=300)   
    plt.show()

    output = net_DAE.get_output(data_tensor.to(device)).cpu().detach().numpy()
    np.save(save_path+task_list[task]+'DAE.npy',np.squeeze(output))