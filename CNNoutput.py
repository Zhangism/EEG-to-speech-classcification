# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:05:23 2020

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
#for task in range(2,5):
model = torch.load(save_path+task_list[task]+'CNN.pkl')
data_np = all_data
data_tensor = torch.from_numpy(data_np).float()
output = model.get_output(data_tensor[0:int(0.5*len(all_data))].to(device)).cpu().detach().numpy()
cnn_data=np.vstack((output,model.get_output(data_tensor[int(0.5*len(all_data)):].to(device)).cpu().detach().numpy()))
np.save(save_path+task_list[task]+'CNN.npy',np.squeeze(cnn_data))