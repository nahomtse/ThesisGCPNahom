pip install igraph
pip install powerlaw
pip install spektral
pip install keras
#################################################################################################################################################
import pdb
import torch
import matplotlib.pyplot as plt
import numpy as np
import igraph
%matplotlib inline
import pandas as pd
import scipy
from utils import graph_from_scores ###
import scipy.io
from scipy.sparse import coo_matrix, isspmatrix
import random

from __future__ import division
import keras
import csv
import tensorflow as tf
from numpy.random import seed
from keras import backend as K
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
seed(1)
tf.random.set_seed(2)
import sklearn.preprocessing as preprocessing
from random import randint
from numpy import array
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding, Input, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from spektral.layers import GCNConv, GlobalSumPool
from spektral.data.loaders import DisjointLoader
from spektral.data import Dataset
from spektral.utils import normalized_laplacian
from spektral.utils import sparse
np.set_printoptions(threshold=sys.maxsize)

import dataset_6_generating_x_and_y 
import loading_xsynthethic_and_ysynthetic_data 
import loading_xtrain_and_ytrain_data

#################################################################################################################################################
def plot_loss_accuracy(history, validation):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    if ('val_categorical_accuracy' in history.history.keys()):
        plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy exp5 ('+str(bits1)+','+str(bits2)+','+str(bits3)+')'',n='+str(n)+',n_batch='+str(n_batch)+',n_epoch='+str(n_epoch)+validation)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('exp5/exp5_Accuracy_('+str(bits1)+','+str(bits2)+','+str(bits3)+')''_n='+str(n)+',n_batch='+str(n_batch)+',n_epoch='+str(n_epoch)+',training=all', format='eps'+validation)
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    if ('val_loss' in history.history.keys()):
        plt.plot(history.history['val_loss'])
    plt.title('model loss exp5 ('+str(bits1)+','+str(bits2)+','+str(bits3)+')'',n='+str(n)+',n_batch='+str(n_batch)+',n_epoch='+str(n_epoch)+validation)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('exp5/Loss_exp5('+str(bits1)+','+str(bits2)+','+str(bits3)+')''_n='+str(n)+',n_batch='+str(n_batch)+',n_epoch='+str(n_epoch)+',training=all', format='eps'+validation)
    plt.show() #synth3005
    
#################################################################################################################################################
#Use this Input format for direct input to LSTM
seqsize = 30
seq_inputs = layers.Input(shape=(seqsize,30,), dtype='float32')
bits1 = 512 #1024
bits2 = 512 #1024
bits3 = 512 #1024
n_epoch= 20 #10
n_batch= 64  #64

# #2 layer LSTM 
# encoder = layers.LSTM(bits1, return_sequences=True, name="lstm_1")(seq_inputs) #1024
# decoder = layers.LSTM(bits2, return_sequences=True, name="lstm_2")(encoder)  #1024
# decoderoutputs = layers.TimeDistributed(layers.Dense(31,activation="softmax"))(decoder)

# #1layer LSTM
# encoder = layers.LSTM(bits1, return_sequences=True, name="lstm_1")(seq_inputs) #1024
# decoderoutputs = layers.TimeDistributed(layers.Dense(31,activation="softmax"))(encoder)

# # 3 layer lstm
encoder = layers.LSTM(bits1, return_sequences=True, name="lstm_1")(seq_inputs) #1024
encoder = layers.LSTM(bits2, return_sequences=True, name="lstm_2")(encoder)  #1024
decoder = layers.LSTM(bits3, return_sequences=True, name="lstm_3")(encoder)  #1024
decoderoutputs = layers.TimeDistributed(layers.Dense(31,activation="softmax"))(decoder)

# 3layer biLSTM
# encoder = layers.Bidirectional(layers.LSTM(bits1, return_sequences=True, name="Bilstm_1"))(seq_inputs)
# encoder = layers.Bidirectional(layers.LSTM(bits2, return_sequences=True, name="Bilstm_2"))(encoder)
# encoder = layers.Bidirectional(layers.LSTM(bits3, return_sequences=True, name="Bilstm_3"))(encoder)
# decoderoutputs = layers.TimeDistributed(layers.Dense(31, activation="softmax"))(encoder)

# # 3 layer GRU
# encoder = layers.GRU(bits1, return_sequences=True, name="GRU_1")(seq_inputs) #1024
# encoder = layers.GRU(bits2, return_sequences=True, name="GRU_2")(encoder)  #1024
# decoder = layers.GRU(bits3, return_sequences=True, name="GRU_3")(encoder)  #1024
# decoderoutputs = layers.TimeDistributed(layers.Dense(31,activation="softmax"))(decoder)

model = tf.keras.Model(inputs=seq_inputs, outputs=decoderoutputs)

# model.compile(optimizer="adam", loss="mean_absolute_percentage_error", metrics=['mean_absolute_percentage_error'])
# model.compile(optimizer="adam", loss="mean_squared_error", metrics=['MeanAbsoluteError'])
# model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_crossentropy'])
print (model.summary())
#################################################################################################################################################


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta = 0.005)
history = model.fit(x_train_15000, y_train_15000, epochs=n_epoch, batch_size=n_batch, verbose=1, shuffle=True,callbacks = [callback] ,validation_data=(xtest_DAS_30,ytest_DAS_30))
plot_loss_accuracy(history, validation = ',DasMAPE')

import json
# Get the dictionary containing each metric and the loss for each epoch

history_dict = history.history
# Save it under the form of a json file
json.dump(history_dict, open('/content/drive/MyDrive/Githubmiller_Latest/netgan_pytorch/netgan/LSTM_MODELS/HISTORY_exp5('+str(bits1)+','+str(bits2)+','+str(bits3)+'),n_batch='+str(n_batch)+',n_epoch='+str(n_epoch)+'Das,training=all''_n='+str(n), 'w'))

history_dict = json.load(open('/content/drive/MyDrive/Githubmiller_Latest/netgan_pytorch/netgan/LSTM_MODELS/HISTORY_exp5('+str(bits1)+','+str(bits2)+','+str(bits3)+'),n_batch='+str(n_batch)+',n_epoch='+str(n_epoch)+'Das,training=all''_n='+str(n), 'r'))

model.save('/content/drive/MyDrive/Githubmiller_Latest/netgan_pytorch/netgan/LSTM_MODELS/exp5('+str(bits1)+','+str(bits2)+','+str(bits3)+')''_n='+str(n)+',n_batch='+str(n_batch)+',n_epoch='+str(n_epoch)+'Das,training=all')

#################################################################################################################################################
def post_process (x2_pred, predicted):
    #Calculate the number of edges which will require correction
    invCols = 0
    edges = 0
    for i in range(x2_pred.shape[0]):
        for j in range(seqsize):
            for k in range(j):
                adj = (x2_pred[i][j][k])
                if ( adj == 1 ):
                    edges += 1
                    if ( int(np.argmax(predicted[i][j])) == int(np.argmax(predicted[i][k])) ) :
                        invCols += 1
                        
    print('Total No of edges ', edges)
    print('No of edges with invalid coloring ', invCols)
    print('Total percentage of edges with invalid colors ', invCols/edges)

  

def post_process_chromatic (x2_pred, predicted):  
    invCols = 0
    edges = 0
    colors_list_list = []
    #for i in range(x2_pred.shape[0]):
    colors_list = []
    for j in range(seqsize):
      for k in range(j):
            if (x2_pred[0][j][k] != 0):
                colors_list.append(np.argmax(predicted[0][j]))
    print('Colors list of graph ', i, ' is  \n', colors_list)
    chromatic_number = len(set(colors_list))
    print('Chromatic number of graph ', i, ' is  ', chromatic_number)
    colors_list_list.append(colors_list)
    return colors_list_list

csv_rows = []
def create_csv_rows (graph_name, colors_list_list_before_correction, colors_list_list_after_correction):    
    for i in range(len(colors_list_list_before_correction)):
        row = [graph_name, i, len(set(colors_list_list_before_correction[i])), len(set(colors_list_list_after_correction[i]))]        
        csv_rows.append(row)
    #print(csv_rows)    

def post_process_correction (x2_pred, predicted, colors_list_list): 
  totInvCols = 0
  totEdges = 0

  for i in range(x2_pred.shape[0]):
      #maxcol = max(xpredicted[i])
      maxcol = max(colors_list_list[i])
      #print(maxcol)
      #mcol = maxcol[0]
      maxorigcol = maxcol
      mcolnew = maxcol
      #print('Maxcol = ',maxcol[0])
      #print(' ... FOR SAMPLE  ... ', i)
      invCols = 0
      edges = 0;
      newCol = 500

      for j in range(seqsize):
          #print(' ... ... FOR EACH NODE ... ...', j)
          for k in range(j):
              #print(' ... ... ... for each adjacency  ... ... ...', k)
              adj = x2_pred[i][j][k]
              #There is an edge
              if ( adj == 1 ):
                  edges += 1
                  if ( np.argmax(predicted[i][j]) == np.argmax(predicted[i][k]) ):                   
                      col_j = np.argmax(predicted[i][j])
                      col_k = np.argmax(predicted[i][k])
                      invCols += 1

                      #Check whether we can give one of the existing colors
                      foundfinalcol = 0
                      for  y in range(1,maxcol+1):
                          #print('Check for COLOR NO ... ', y)
                          if ( foundfinalcol == 1 ) :
                              #print('FOUND COLOR ALREADY  ... leave the loop')
                              break

                          foundcol = 0
                          #Check the adjacent nodes of j
                          #for  z in range(j):
                          for z in range(seqsize):
                              if j!=z:
                                  if  (   ((x2_pred[i][j][z] == 1) and (np.argmax(predicted[i][z]) == y))
                                      or  ((x2_pred[i][z][j] == 1) and (np.argmax(predicted[i][z]) == y))
                                      ):
                                      #print('[1] Adjacent node ... from ',j, '-->', z, 'color = ',xpredicted[i][z][0] )
                                      foundcol = 1
                                      #print('[1] Found Color ', y, ' for node ', z, 'from node ', j )
                                      break

                          #Finished checking the adjacent nodes of j
                          #Color y is not used by any of j's neighbours
                          #print('[1] Finished Checking the adjacent node of ... ',j,' ... foundcol = ',foundcol)
                          if ( foundcol == 0 ) :
                              #assign any prediction > 1
                              predicted[i][j][y] = 2
                              #print('[1] Reuse color ', y, ' for node ', j)
                              foundfinalcol = 1

                          else :
                              foundcol = 0                                                            
                              #Check the adjacent nodes of k
                              for z in range(seqsize):
                                  if k!=z:
                                      if  (   ((x2_pred[i][k][z] == 1) and (np.argmax(predicted[i][z]) == y))
                                          or  ((x2_pred[i][z][k] == 1) and (np.argmax(predicted[i][z]) == y))
                                          ):
                                          #print('[1] Adjacent node ... from ',j, '-->', z, 'color = ',xpredicted[i][z][0] )
                                          foundcol = 1
                                          #print('[1] Found Color ', y, ' for node ', z, 'from node ', j )
                                          break
                              #Color y is not used by any of k's neighbours
                              if ( foundcol == 0 ) :
                                  #assign any prediction > 1
                                  predicted[i][k][y] = 2
                                  #print('[2] Reuse color ', y, ' for node ', k )
                                  foundfinalcol = 1

                      # Could not color using an existing color
                      # Get a new color from 500 onwards OR use from the new 500 color number series
                      if ( foundfinalcol == 0 ) :
                           #newCol += 1
                           mcolnew += 1
                           #assign any prediction > 1
                           predicted[i][k][mcolnew] = 2
                           maxcol +=1
                           #print('Use new color ', mcolnew, ' for node ', k)

  return predicted
#################################################################################################################################################
model.evaluate(xtest_DAS_30.astype('float32'), ytest_DAS_30.astype('float32'),batch_size=n_batch, verbose=1,callbacks = [callback])
#################################################################################################################################################
seqsize = n

for i,csv in enumerate(xtest):
  print('\n------PREDICTING ',xtest.shape,'-------')
  predicted = model.predict(xtest[i], batch_size=None, verbose=0, steps=None, callbacks=None)        
  print('\nInvalid edges percentage before color correction ->')
  print(post_process (xtest[i], predicted))
  print('\nColors list and Chromatic number predicted by the model ->')
  colors_list_list = post_process_chromatic(np.asarray(xtest[i]), predicted)
  # print(colors_list_list)
  print('\nApply color correction ->')
  print('\nApply color correction ->')
  predicted = post_process_correction(np.asarray(xtest[i]), predicted, colors_list_list)
  print('\nColors list and Chromatic number following color correction ->')
  colors_list_list = post_process_chromatic(np.asarray(xtest[i]), predicted)
  # print(colors_list_list)
  print('\nInvalid edges percentage after color correction ->')
  print(post_process(np.asarray(xtest[i]), predicted))
  print('--------------END OF PREDICTION -----------------------' )
#################################################################################################################################################
reshapert = x_synth_totaal.astype('float32')
for i in range(5):
  first = nx.from_numpy_array(reshapert[i])
  color_map = varY3_test[i]
  nx.draw(first, node_size=500, alpha=1, node_color =color_map)
  plt.show()