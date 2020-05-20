# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:51:36 2020

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
NET = 'Leaky_Dropout'
subject = 0
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda")

save_path = 'Q:\\大学\\毕业设计\\代码\\'
FEATURE = 'CCM'
NET = ''
subject = 0
os.environ['CUDA_VISIBLE_DEVICES']='0'
 
'''
1: Vowel only (0) vs consonant (1)
2: non-nasal (0) vs nasal (1)
3: non-bilabial (0) vs bilabial (1)
4: non-iy (0) vs iy (1)
5: non-uw (0) vs uw (1)
'''
task = '0'
device = torch.device("cuda")
all_data = np.load(save_path+'CCM.npy', allow_pickle=True)
data = np.zeros((len(all_data),1,62,62))#1913,62,62
for i in range(0,len(all_data)):
    data[i][0]= all_data[i][0]
label_class = [np.array([0,0,0,0,1]),np.array([1,0,0,1,0]),np.array([1,0,1,1,0]),np.array([0,0,0,1,0]),np.array([1,1,1,0,0]),np.array([1,1,0,0,0]),np.array([1,0,0,1,0]),np.array([1,0,1,0,0]),np.array([1,0,1,0,0]),np.array([1,1,0,0,0]),np.array([1,1,0,0,0])]
labels = all_data[:,1]
labels_binary = np.zeros((len(all_data),5))
for i in range(0,len(all_data)):
    labels_binary[i] = label_class[int(labels[i])]
labels_binary_np = np.array(labels_binary)
for task in range(0,5):
    label = labels_binary_np[:,task]
    np.save(save_path+'class'+str(task)+'.npy',label)

np.save(save_path+'class_data.npy',data)
'''
training_data_np = data[0:int(0.8*len(data))]
training_label_np = label[0:int(0.8*len(data))]
training_data_tensor = torch.from_numpy(training_data_np).float()
training_labels_tensor = torch.squeeze(torch.from_numpy(training_label_np.astype(np.float)).long())    
training_set = TensorDataset(training_data_tensor,training_labels_tensor)
train_loader = DataLoader(dataset = training_set, batch_size = 64, shuffle = True, num_workers = 0)

testing_data_np = data[int(0.8*len(data)):]
testing_label_np = label[int(0.8*len(data)):]
testing_data_tensor = torch.from_numpy(testing_data_np).float()
testing_labels_tensor = torch.squeeze(torch.from_numpy(testing_label_np.astype(np.float)).long())    
testing_set = TensorDataset(testing_data_tensor,testing_labels_tensor)
test_loader = DataLoader(dataset = testing_set, batch_size = 1, shuffle = True, num_workers = 0)'''