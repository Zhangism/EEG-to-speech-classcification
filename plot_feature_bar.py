# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:38:18 2020

@author: zhang
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

mfcc = [21.21212121, 25.92592593, 18.51851852, 37.03703704, 18.51851852,
       25.92592593, 14.81481481, 22.22222222, 18.51851852, 22.22222222,
       25.92592593]
linear = [24.24242424, 22.22222222, 14.81481481, 18.51851852, 14.81481481,
       18.51851852, 18.51851852, 25.92592593, 22.22222222, 11.11111111,
       18.51851852]
wavelet = [18.18181818, 14.81481481, 18.51851852, 37.03703704, 29.62962963,
       18.51851852, 22.22222222, 11.11111111, 25.92592593, 14.81481481,
       37.03703704]
labels = [1,2,3,4,5,6,7,8,9,10,11]
x = np.arange(len(mfcc))
width = 0.3
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, mfcc, width, label='MFCC')
rects2 = ax.bar(x + width, linear, width, label='Linear')
rects3 = ax.bar(x        , wavelet, width, label='Wavelet')

ax.set_ylabel('Accuracy(%)')
ax.set_xlabel('Particitants')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


fig.tight_layout()
plt.savefig("Q:\\大学\\毕业设计\\图片\\网络结构比较\\"+'Acc over all'+'.jpg',dpi=300)    
plt.show()