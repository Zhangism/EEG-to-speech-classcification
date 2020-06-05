# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:47:40 2020

@author: zhang
"""
import os
import numpy as np
from scipy.io import loadmat
import random
path_mat = '\\new\\thinkeing.mat'
path = 'A:\\MM\\'
save_path = 'Q:\\大学\\毕业设计\\代码\\'
dirs = os.listdir(path)
X = []
for file in dirs:
    #提取数据
    dataMat=loadmat(path+file+'\\new\\thinking.mat')
    array = dataMat['thinking_mats']
    X.append(array)
    print(file)
CCM = []
for i in range(0,len(X)):
    temp = []
    for j in range(0,X[i][0].shape[0]):
        temp.append(np.cov(X[i][0][j]))
    CCM.append(temp)

labels = np.empty([0])#所有的标签
Dict = {'/uw/': 0,'/tiy/': 1, '/piy/': 2,'/iy/': 3,'/m/': 4,'/n/': 5,'/diy/': 6,'pat': 7,'pot': 8,'gnaw': 9,'knew': 10}
dirs = os.listdir(path)
labels = np.empty([0])#所有的标签
L = []
for file in dirs:
    Y = np.empty([0])#一个文件的所有行
    count = 0
    with open(path+file+"\\kinect_data\\labels.txt", "r") as f:#提取标签
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            Y = np.append(Y,Dict[line])
            count = count + 1
    L.append(Y)
labels = np.array(L)

all_data = []
for i in range(0,len(CCM)):
    for j in range(0,len(CCM[i])):
        all_data.append((CCM[i][j],labels[i][j]))

#打乱
random.shuffle(all_data)
np.save(save_path+'CCM.npy',np.squeeze(all_data))