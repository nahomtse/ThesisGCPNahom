import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import pandas as pd
import scipy
import random
###################################################################################################################################################
#erdos training (dataset 1)  
xtrain_1000_n30_erdos_p01_1 = np.load('trainset/xtrain_2000_n30_erdos_p01_1.npy')
ytrain_1000_n30_erdos_p01_1 = np.load('trainset/ytrain_2000_n30_erdos_p01_1.npy')

xtrain_1000_n30_erdos_p12_1 = np.load('trainset/xtrain_2000_n30_erdos_p12_1.npy')
ytrain_1000_n30_erdos_p12_1 = np.load('trainset/ytrain_2000_n30_erdos_p12_1.npy')

xtrain_1000_n30_erdos_p23_1 = np.load('trainset/xtrain_2000_n30_erdos_p23_1.npy')
ytrain_1000_n30_erdos_p23_1 = np.load('trainset/ytrain_2000_n30_erdos_p23_1.npy')

xtrain_1000_n30_erdos_p34_1 = np.load('trainset/xtrain_2000_n30_erdos_p34_1.npy')
ytrain_1000_n30_erdos_p34_1 = np.load('trainset/ytrain_2000_n30_erdos_p34_1.npy')

xtrain_1000_n30_erdos_p45_1 = np.load('trainset/xtrain_2000_n30_erdos_p45_1.npy')
ytrain_1000_n30_erdos_p45_1 = np.load('trainset/ytrain_2000_n30_erdos_p45_1.npy')

x_erdos_combined = np.concatenate((xtrain_1000_n30_erdos_p01_1,
                                   xtrain_1000_n30_erdos_p12_1,
                                   xtrain_1000_n30_erdos_p23_1,
                                   xtrain_1000_n30_erdos_p34_1, ##
                                   xtrain_1000_n30_erdos_p45_1), axis = 0)

y_erdos_combined = np.concatenate((ytrain_1000_n30_erdos_p01_1,
                                   ytrain_1000_n30_erdos_p12_1,
                                   ytrain_1000_n30_erdos_p23_1,
                                   ytrain_1000_n30_erdos_p34_1,
                                   ytrain_1000_n30_erdos_p45_1), axis = 0)

print(x_erdos_combined.shape)
print(y_erdos_combined.shape)

###################################################################################################################################################
#wattss training (dataset 2)
xtrain_1000_n30_watts_p01_1 = np.load('trainset/xtrain_5000_n30_watts_p01_1.npy')
ytrain_1000_n30_watts_p01_1 = np.load('trainset/ytrain_5000_n30_watts_p01_1.npy')

xtrain_1000_n30_watts_p12_1 = np.load('trainset/xtrain_5000_n30_watts_p12_1.npy')
ytrain_1000_n30_watts_p12_1 = np.load('trainset/ytrain_5000_n30_watts_p12_1.npy')

xtrain_1000_n30_watts_p23_1 = np.load('trainset/xtrain_5000_n30_watts_p23_1.npy')
ytrain_1000_n30_watts_p23_1 = np.load('trainset/ytrain_5000_n30_watts_p23_1.npy')

xtrain_1000_n30_watts_p34_1 = np.load('trainset/xtrain_5000_n30_watts_p34_1.npy')
ytrain_1000_n30_watts_p34_1 = np.load('trainset/ytrain_5000_n30_watts_p34_1.npy')

xtrain_1000_n30_watts_p45_1 = np.load('trainset/xtrain_5000_n30_watts_p45_1.npy')
ytrain_1000_n30_watts_p45_1 = np.load('trainset/ytrain_5000_n30_watts_p45_1.npy')


x_watts_combined = np.concatenate((xtrain_1000_n30_watts_p01_1,#
                                   xtrain_1000_n30_watts_p12_1,
                                   xtrain_1000_n30_watts_p23_1,#
                                   xtrain_1000_n30_watts_p34_1,#
                                   xtrain_1000_n30_watts_p45_1), axis = 0)#

y_watts_combined = np.concatenate((ytrain_1000_n30_watts_p01_1,#
                                   ytrain_1000_n30_watts_p12_1,
                                   ytrain_1000_n30_watts_p23_1,#
                                  ytrain_1000_n30_watts_p34_1,#
                                   ytrain_1000_n30_watts_p45_1), axis = 0)#

print(x_watts_combined.shape)
print(y_watts_combined.shape)
###################################################################################################################################################
#barabasi  (dataset 3)
xtrain_1000_n30_basi_p01_1 = np.load('trainset/xtrain_5000_n30_basi_p01_1.npy')
ytrain_1000_n30_basi_p01_1 = np.load('trainset/ytrain_5000_n30_basi_p01_1.npy')

xtrain_1000_n30_basi_p12_1 = np.load('trainset/xtrain_5000_n30_basi_p12_1.npy')
ytrain_1000_n30_basi_p12_1 = np.load('trainset/ytrain_5000_n30_basi_p12_1.npy')

xtrain_1000_n30_basi_p23_1 = np.load('trainset/xtrain_5000_n30_basi_p23_1.npy')
ytrain_1000_n30_basi_p23_1 = np.load('trainset/ytrain_5000_n30_basi_p23_1.npy')

xtrain_1000_n30_basi_p34_1 = np.load('trainset/xtrain_5000_n30_basi_p34_1.npy')
ytrain_1000_n30_basi_p34_1 = np.load('trainset/ytrain_5000_n30_basi_p34_1.npy')

xtrain_1000_n30_basi_p45_1 = np.load('trainset/xtrain_5000_n30_basi_p45_1.npy')
ytrain_1000_n30_basi_p45_1 = np.load('trainset/ytrain_5000_n30_basi_p45_1.npy')

x_basi_combined = np.concatenate((xtrain_1000_n30_basi_p01_1,
                                   xtrain_1000_n30_basi_p12_1, #
                                   xtrain_1000_n30_basi_p23_1,
                                   xtrain_1000_n30_basi_p34_1,
                                   xtrain_1000_n30_basi_p45_1), axis = 0)

y_basi_combined = np.concatenate((ytrain_1000_n30_basi_p01_1,
                                  ytrain_1000_n30_basi_p12_1, # 
                                   ytrain_1000_n30_basi_p23_1,
                                   ytrain_1000_n30_basi_p34_1,
                                   ytrain_1000_n30_basi_p45_1), axis = 0)

print(x_basi_combined.shape)
print(y_basi_combined.shape)

###################################################################################################################################################
#totaal training set (Dataset 4)
x_train_15000 = np.concatenate((x_erdos_combined, x_watts_combined, x_basi_combined), axis = 0 )
y_train_15000 = np.concatenate((y_erdos_combined, y_watts_combined, y_basi_combined), axis = 0 )

np.save('trainset/x_train_totaalall.npy',x_train_15000)
np.save('trainset/y_train_totaalall.npy',y_train_15000)

x_train_15000 = np.load('trainset/x_train_totaalall.npy')
y_train_15000 = np.load('trainset/y_train_totaalall.npy')

print(x_train_15000.shape)
print(y_train_15000.shape)