# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:17:45 2020

@author: zhang
"""
from scipy.io import loadmat
import os
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pywt
from pywt import wavedec
import librosa
import random
FEATURE = 'MFCC'
def feature_exraction(array,feature):
    length = array.shape[1]
    epochs = array[0]
    epoch_wd = np.zeros([length,17,62,500])
    count = 0
    for epoch in epochs:#epoch.shape = 62x4900
        #print(epoch.shape)
        for i in range(0,17):
            epoch_wd[count][i]=epoch[...,0+i*250:500+i*250]
            #print(epoch_wd[0][i][0])
        count=count+1
    X = epoch_wd    
    if feature == 'Wavelet':
        db4 = pywt.Wavelet('db4')
        X_wave = np.zeros([length,17,62,5])
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
        return X_wave
    elif feature == 'MFCC':
        X_MFCC = np.zeros([length,17,62,13])
        for i in range(0,X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    X_MFCC[i][j][k] = np.squeeze(librosa.feature.mfcc(y=X[i][j][k], sr=1000,n_mfcc=13))      
        return X_MFCC
    elif feature == 'Linear':
        X_linear = np.zeros([length,17,62,12])
        for i in range(0,X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    X_linear[i][j][k][0] = np.mean(X[i][j][k])#均值
                    X_linear[i][j][k][1] = np.mean(np.fabs(X[i][j][k]))#绝对值的均值
                    X_linear[i][j][k][2] = np.std(X[i][j][k])#标准差
                    X_linear[i][j][k][3] = np.sum(X[i][j][k])#和
                    X_linear[i][j][k][4] = np.median(X[i][j][k])#中位数
                    X_linear[i][j][k][5] = np.var(X[i][j][k])#方差
                    X_linear[i][j][k][6] = np.max(X[i][j][k])#最大值
                    X_linear[i][j][k][7] = np.max(np.fabs(X[i][j][k]))#绝对值的最大值
                    X_linear[i][j][k][8] = np.min(X[i][j][k])#最小值
                    X_linear[i][j][k][9] = np.min(np.fabs(X[i][j][k]))#绝对值的最小值
                    X_linear[i][j][k][10] = np.max(X[i][j][k]) + np.min(X[i][j][k])#最大值加最小值
                    X_linear[i][j][k][11] = np.max(X[i][j][k]) - np.min(X[i][j][k])#最大值减最小值
        return X_linear
path_mat = '\\new\\thinkeing.mat'
path_label = '\\kinect_data\\labels.txt'
path = 'A:\\MM\\'
save_path = 'Q:\\大学\\毕业设计\\代码\\'
dict = {'/uw/': 0,'/tiy/': 1, '/piy/': 2,'/iy/': 3,'/m/': 4,'/n/': 5,'/diy/': 6,'pat': 7,'pot': 8,'gnaw': 9,'knew': 10}
dirs = os.listdir(path)
labels = np.empty([0])#所有的标签
L = []
for file in dirs:
    Y = np.empty([0])#一个文件的所有行
    count = 0
    with open(path+file+"\\kinect_data\\labels.txt", "r") as f:#提取标签
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            Y = np.append(Y,dict[line])
            count = count + 1
    L.append(Y)
labels = np.array(L)
    #temp = np.hstack((labels,Y))
    #labels = np.empty([temp.size])
    #labels = temp
X = []
for file in dirs:
    #提取数据
    dataMat=loadmat(path+file+'\\new\\thinking.mat')
    array = dataMat['thinking_mats']
    X.append(feature_exraction(array,FEATURE))
    print(file)
data = np.array(X)
'''for i in range(0,len(X)):
    if i == 0:
        data = X[0]
    else:
        data = np.vstack((data,X[i]))'''

#数据标签合并后打乱
all_data = []
for i in range(0,data.shape[0]):
    temp = []
    for j in range(0,data[i].shape[0]):
        temp.append([data[i][j],labels[i][j]])
    random.shuffle(temp)
    all_data.append(np.array(temp))
    #all_data.append([(data[i],labels[i])])

#打乱
#random.shuffle(all_data)
np.save(save_path+FEATURE+'data'+'.npy',np.squeeze(all_data))