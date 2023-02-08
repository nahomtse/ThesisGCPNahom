# pip install igraph
# pip install powerlaw
# pip install gurobipy

import training 
from training import Trainer 
import utils 
import pdb
import torch
import matplotlib.pyplot as plt
import numpy as np
import igraph
%matplotlib inline
import pandas as pd
import scipy
from utils import graph_from_scores ###
import networkx as nx
import scipy.io
from scipy.sparse import coo_matrix, isspmatrix

import gurobipy as gp
from gurobipy import *

import tensorflow as tf
import random
######################################################################################
###################################################################################################################################################
# create test graph (THIS IS AN EXAMPLE)
n_test = 30
p_test = 0.6
m_test = 27 ###########################################################################################################
q_test = 12
#G_test = nx.erdos_renyi_graph(n_test, p_test)
#G_test = nx.barabasi_albert_graph(n_test,m_test)
# G_test = nx.barabasi_albert_graph(n_test,m_test)
G_test = nx.watts_strogatz_graph(n_test,q_test,p_test)


graph = nx.to_numpy_array(G_test)
#graph = np.pad(graph, (35,35) , mode= 'edge')
graph[graph!=0] = 1.0
graph_nx = nx.from_numpy_array(graph)
graph_sparse = scipy.sparse.csr_matrix(graph)

n_edges = graph.sum()

nx.draw(graph_nx, with_labels = True)
graph_sparse
###################################################################################################################################################
trainer = Trainer(graph_sparse, len(graph), max_iterations=20000, rw_len=6, batch_size=128, H_gen=40, H_disc=30, H_inp=128, z_dim=16, lr=0.0003,
                  n_critic=3, gp_weight=10.0, betas=(.5, .9), l2_penalty_disc=5e-5, l2_penalty_gen=1e-7, temp_start=5.0,  
                  val_share=0.2, test_share=0.1, seed=20, set_ops=False) #max iterationn = 20000

trainer.train(create_graph_every=100, plot_graph_every=200, num_samples_graph=50000, stopping_criterion='val')
###################################################################################################################################################
trans_mat = trainer.create_transition_matrix(50000)
graph_synthetic = []
for i in range(100):
    graph_sampled = graph_from_scores(trans_mat, n_edges)
    graph_synthetic.append(graph_sampled)
    graph_nx_sampled = nx.from_numpy_array(graph_sampled)
    nx.draw(graph_nx_sampled, node_size=25, alpha=0.5)
    plt.show()


for i, graph_sampled in enumerate(graph_synthetic):
    graph_sampled = scipy.sparse.csc_matrix(graph_sampled)
    path = 'graph_eigendata_test(n=30)_basi_30_12'+str(i) + '.npz'
    scipy.sparse.save_npz(path, graph_sampled)
###################################################################################################################################################
graph_sampled = scipy.sparse.csr_matrix.toarray(graph_sampled)
t2 = np.array(graph_synthetic)
print(t2.shape)

fake_graph = np.reshape(graph_sampled, (1,) + (graph_sampled.shape))
x2_pred = fake_graph
test_B = nx.from_numpy_array(graph_sampled)

###################################################################################################################################################
varY_list_test = []
varY2_list_test = []
werk = 5
ranlist = []
ran = []
n = 30 
lengte = []
t2 = reshapert
for p in range(werk):
  test_B = nx.from_numpy_array(t2[p])
  m = gp.Model('chrom_num', env =e)
  
  # get maximum number of variables necessary
  k = max(dict(nx.degree(test_B)).values()) + 1
  K= range(k)
  #################################################################################################################################
  # create k binary variables, y_0 ... y_{k-1} to indicate whether color k is used
  y = []
  for j in range(k):
    y.append(m.addVar(vtype=gp.GRB.BINARY, name='y_%d' % j, obj=1))

  # create n * k binary variables, x_{l,j} that is 1 if node l is colored with j
  x = []
  for l in range(n):
    x.append([])
    for j in range(k):
      x[-1].append(m.addVar(vtype=gp.GRB.BINARY, name='x_%d_%d' % (l, j), obj=0))


  # objective function is minimize colors used --> sum of y_0 ... y_{k-1}
  m.setObjective(gp.quicksum(y[j] for j in K), gp.GRB.MINIMIZE)
  m.update()

  # add constraint -- each node gets exactly one color (sum of colors used is 1)
  for u in range(n):
      m.addConstr(gp.quicksum(x[u]) == 1, name='NC_%d')

  # add constraint -- keep track of colors used (y_j is set high if any time j is used)
  for l in range(n):
      for j in range(k):
          m.addConstr(x[u][j] <= y[j], name='SH_%d_%d')

  # add constraint -- adjacent nodes have different colors
  for u in range(n):
      for v in test_B[u]:
          if v > u:
              for j in range(k):
                  m.addConstr(x[u][j] + x[v][j] <= 1, name='ADJ_%d_%d_COL_%d')

  # add constraint -- adjacent nodes have different colors
  for u in range(n):
      for v in test_B[u]:
          if v > u:
              for j in range(k):
                  m.addConstr(x[u][j] + x[v][j] <= y[j], name='ADJ_%d_%d_COL_%d')
  #################################################################################################################################
  # update model, solve, return the chromatic number

  m.update()
  m.setParam('MIPGap', 0.01)
  m.setParam('TimeLimit', 80)
  m.optimize()
  #chrom_num = []

  chrom_num_test = m.objVal
  chrom_num_int_test = int(chrom_num_test)
  print('-------')
  print('-------')
  print("chromatic number of this graph is ", m.objVal)

###################################################################################################################################################
  varZ_test= m.getAttr("x")
  varX_test = m.getVars()  
  lengte.append(len(y))
  vargurY_test = varZ_test[0:lengte[p]]
  vargurX_test = varZ_test[lengte[p]:] 

  varY3_reshape_test = np.reshape(vargurX_test,(n,lengte[p]))
  varY3_backup_test = varY3_reshape_test
  # print('backupVarY3', varY3_backup)
  for ti in range(chrom_num_int_test):
    varY3_reshape_test[:,ti] = varY3_reshape_test[:,ti]*(ti+1)
    varY3_reshape_new_test = np.extract(varY3_reshape_test !=0, varY3_reshape_test)
    varY3_reshape_list_test = varY3_reshape_new_test.tolist()
  # print('de gereshapte versie', varY3_reshape)
  # print('alleen de kleuren', varY3_reshape_new)
  # y2_train.shape
  varY_list_test.append(varY3_reshape_list_test)
  varY2_list_test.append(list(varY_list_test))
  varY3_test = np.asarray(varY2_list_test[p])
#   y2_train_test = tf.keras.utils.to_categorical(y = varY3_test, num_classes=n+1)

  m.reset()
###################################################################################################################################################  
np.save('testset/xtest_5_faken30.npy', fake_graph)
np.save('testset/ytest_5_faken30.npy', y2_train_test)
x_test =  np.load('testset/xtest_5_faken30.npy')
y_test = np.load('testset/ytest_5_faken30.npy')