# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:22:36 2020

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
save_path = 'Q:\\大学\\毕业设计\\代码\\'
FEATURE = 'CCM'
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda")
task = 4
all_data = np.load(save_path+'class_data.npy',allow_pickle=True)
'''
label_class = [np.array([0,0,0,0,1]),np.array([1,0,0,1,0]),np.array([1,0,1,1,0]),np.array([0,0,0,1,0]),np.array([1,1,1,0,0]),np.array([1,1,0,0,0]),np.array([1,0,0,1,0]),np.array([1,0,1,0,0]),np.array([1,0,1,0,0]),np.array([1,1,0,0,0]),np.array([1,1,0,0,0])]
labels = []
for i in range(0,5):
    labels.append(np.load(save_path+'class'+str(i)+'.npy',allow_pickle=True))
label_np = labels[task]
labels_tensor = torch.squeeze(torch.from_numpy(label_np.astype(np.float)).long())
'''
task_list = ['Vowel only vs consonant','non-nasal vs nasal', 'non-bilabial vs bilabial ','non-iy vs iy ','non-uw vs uw']


'''
1: Vowel only (0) vs consonant (1)
2: non-nasal (0) vs nasal (1)
3: non-bilabial (0) vs bilabial (1)
4: non-iy (0) vs iy (1)
5: non-uw (0) vs uw (1)
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm1 = nn.LSTM(62, 128,batch_first=True)
        self.lstm2 = nn.LSTM(128, 256,batch_first=True)
        self.classifier1 = nn.Linear(256, 512)
        self.dp1 = nn.Dropout(p=0.25)
        self.classifier2 = nn.Linear(512, 1024)
        self.dp2 = nn.Dropout(p=0.5)
        self.classifier3 = nn.Linear(1024, 2)
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

#for task in range(2,5):
net = torch.load(save_path+task_list[task]+'LSTM.pkl')
data_np = all_data
data_tensor = torch.squeeze(torch.from_numpy(data_np).float())
output = net.get_output(data_tensor[0:int(0.5*len(all_data))].to(device)).cpu().detach().numpy()
cnn_data=np.vstack((output,net.get_output(data_tensor[int(0.5*len(all_data)):].to(device)).cpu().detach().numpy()))
np.save(save_path+task_list[task]+'LSTM.npy',np.squeeze(cnn_data))