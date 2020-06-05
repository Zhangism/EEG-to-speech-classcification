# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:48:17 2020

@author: zhang
"""
import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from xgboost import plot_importance #显示特征重要性
from matplotlib import pyplot
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
label_class = [np.array([0,0,0,0,1]),np.array([1,0,0,1,0]),np.array([1,0,1,1,0]),np.array([0,0,0,1,0]),np.array([1,1,1,0,0]),np.array([1,1,0,0,0]),np.array([1,0,0,1,0]),np.array([1,0,1,0,0]),np.array([1,0,1,0,0]),np.array([1,1,0,0,0]),np.array([1,1,0,0,0])]
labels = []
accuracy = []
param = []
save_path = 'Q:\\大学\\毕业设计\\代码\\'
for i in range(0,5):
    labels.append(np.load(save_path+'class'+str(i)+'.npy',allow_pickle=True))
task_list = ['Vowel only vs consonant','non-nasal vs nasal', 'non-bilabial vs bilabial ','non-iy vs iy ','non-uw vs uw']

parameters = {
    'max_depth': [10],
    'learning_rate': [ 0.1],
    'n_estimators': [5000],
    'min_child_weight': [ 2],
    'max_delta_step': [0.3],
    'subsample': [0.8],
    'colsample_bytree': [0.4,0.7],
    'reg_alpha': [ 0,0.25]

}
for task in range(0,5):
    all_data = np.load(save_path+task_list[task]+'DAE.npy',allow_pickle=True)
    train_x, valid_x, train_y, valid_y = train_test_split(all_data, labels[task], test_size=0.1, random_state=1)  # 分训练集和验证集
# 这里不需要Dmatrix
    xlf = xgb.XGBClassifier(max_depth=10,
                            learning_rate=0.1,
                            n_estimators=2000,
                            silent=True,
                            objective='binary:logistic',
                            nthread=-1,
                            gamma=0,
                            min_child_weight=1,
                            max_delta_step=0,
                            subsample=0.85,
                            colsample_bytree=0.7,
                            colsample_bylevel=1,
                            reg_alpha=0,
                            reg_lambda=1,
                            scale_pos_weight=1,
                            seed=1440,
                            missing=None)
# 有了gridsearch我们便不需要fit函数
    gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3)
    gsearch.fit(train_x, train_y)
    print("Best score: %0.3f" % gsearch.best_score_)
    accuracy.append(gsearch.best_score_)
    
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    param.append(best_parameters)