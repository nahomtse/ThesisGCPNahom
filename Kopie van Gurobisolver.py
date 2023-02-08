# pip install igraph
# pip install powerlaw
# pip install gurobipy

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
# Create environment with WLS license
e = gp.Env(empty=True)
e.setParam('WLSACCESSID', '5f70bf63-9610-412d-8ca1-551e7e052c61')
e.setParam('WLSSECRET', '190b5abe-a47c-4738-b218-bb7821931920')
e.setParam('LICENSEID', 887511)
#e.setParam('OutputFlag', 0)
e.start()
######################################################################################
varY_list = []
varY2_list = []
lengte = []
werk = 100
ranlist = []
ran = []

# for i in range(werk):
    #Let's say I will have 100 nodes and the connection probability is 0.4
n = 30
# p = 0.2
# piet = np.arange(0,1, (1/werk))
# pietje = piet.tolist()
for p in range(0,werk,1):
    # G = nx.barabasi_albert_graph(n,(random.randint(0, n)))
    #G = nx.barabasi_albert_graph(n,10)
    G= nx.erdos_renyi_graph(n,(p/(werk)))
    #G= nx.erdos_renyi_graph(n,((p+(1*werk))/(werk*5)))
    #G = nx.watts_strogatz_graph(n,10,(p/(werk*10)))
    Gr = nx.to_numpy_array(G)
    
    ranlist.append(Gr)
    ran.append(list(ranlist))
    randy = np.asarray(ran[p])
    xtrain = randy 

    # print(randy.shape)
    G_new = nx.from_numpy_array(randy[p])
    # Create the model within the Gurobi environment

    m = gp.Model('chrom_num', env =e)
    
    # get maximum number of variables necessary
    k = max(dict(nx.degree(G_new)).values()) + 1
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


    # label_dict = {}
    # label_num = 0
    # for j in range(n):
    #       if (y[j] == 0):
    #           y[j] = 0  # so if value of out put is zero, than varY remains zero
              
    #       else:
    #           if (varY[j]) in label_dict:
    #              varY[j] = label_dict[varY[j]]
    #           else:
    #              label_dict[varY[j]] = label_num #if the value is already in the label_dict, no new label/color has to be made
    #              varY[j] = label_num
    #              label_num = label_num + 1  


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
        for v in G[u]:
            if v > u:
                for j in range(k):
                    m.addConstr(x[u][j] + x[v][j] <= 1, name='ADJ_%d_%d_COL_%d')

    # add constraint -- adjacent nodes have different colors
    for u in range(n):
        for v in G[u]:
            if v > u:
                for j in range(k):
                    m.addConstr(x[u][j] + x[v][j] <= y[j], name='ADJ_%d_%d_COL_%d')
#################################################################################################################################
    # update model, solve, return the chromatic number
    
    m.update()
    #m.setParam('MIPGap', 0.05)
    m.setParam('TimeLimit', 15)
    m.optimize()
    #chrom_num = []
    
    chrom_num = m.objVal
    chrom_num_int = int(chrom_num)
    print('-------')
    print('-------')
    print("chromatic number of this graph is ", m.objVal)


    varZ= m.getAttr("x")
    varX = m.getVars()  
    lengte.append(len(y))
    vargurY = varZ[0:lengte[p]]
    vargurX = varZ[lengte[p]:] 

    varY3_reshape = np.reshape(vargurX,(n,lengte[p]))
    varY3_backup = varY3_reshape
    # print('backupVarY3', varY3_backup)
    # with the next loop colors are assigned to each node according to the solver and shaped to size of the graph 
    for ti in range(chrom_num_int):
      varY3_reshape[:,ti] = varY3_reshape[:,ti]*(ti+1)
      varY3_reshape_new = np.extract(varY3_reshape!=0, varY3_reshape)
      varY3_reshape_list = varY3_reshape_new.tolist()
      # varY3_reshape_list = ertg.tolist()
    # print('de gereshapte versie', varY3_reshape)
    # print('alleen de kleuren', varY3_reshape_new)
    # y2_train.shape
    varY_list.append(varY3_reshape_list)
    varY2_list.append(list(varY_list))
    varY3 = np.asarray(varY2_list[p])
    y2_train = tf.keras.utils.to_categorical(y = varY3, num_classes=n+1)

    m.reset()
##########################################################################################################
print('shape van de reshape is',varY3.shape)
print(varY3)
print('n is ', n)
print('loop is ', werk)
print(y2_train.shape)

np.save('testset/xtest_100_erdos.npy', xtrain)
np.save('testset/ytest_100_erdos.npy', y2_train)
xtest_100_erdos = np.load('testset/xtest_100_erdos.npy')
ytest_100_erdos = np.load('testset/ytest_100_erdos.npy')
