import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import pandas as pd
import scipy
import random
###################################################################################################################################################
#erdos synthetic (Dataset 5.1)
# n =30 en p = 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 elk 110 

e9X = np.load('synthetic_set/X_syn_erdos_n30_p0.9.npy')
e8X = np.load('synthetic_set/X_syn_erdos_n30_p0.8.npy')
e7X = np.load('synthetic_set/X_syn_erdos_n30_p0.7.npy')
e6X = np.load('synthetic_set/X_syn_erdos_n30_p0.6.npy')
e5X = np.load('synthetic_set/X_syn_erdos_n30_p0.5.npy')
e4X = np.load('synthetic_set/X_syn_erdos_n30_p0.4.npy')
e3X = np.load('synthetic_set/X_syn_erdos_n30_p0.3.npy')
e2X = np.load('synthetic_set/X_syn_erdos_n30_p0.2.npy')
e1X = np.load('synthetic_set/X_syn_erdos_n30_p0.1.npy')



ptotaalerdosX = np.concatenate((e9X,e8X,e7X,e6X,e5X,e4X,e3X,e2X,e1X), axis = 0 )
print(ptotaalerdosX.shape) #(990,30,30)
np.save('synthetic_set/Xtotalsynt_erdos_n30',ptotaalerdosX)

e9Y = np.load('synthetic_set/Y_syn_erdos_n30_p0.9.npy')
e8Y = np.load('synthetic_set/Y_syn_erdos_n30_p0.8.npy')
e7Y = np.load('synthetic_set/Y_syn_erdos_n30_p0.7.npy')
e6Y = np.load('synthetic_set/Y_syn_erdos_n30_p0.6.npy')
e5Y = np.load('synthetic_set/Y_syn_erdos_n30_p0.5.npy')
e4Y = np.load('synthetic_set/Y_syn_erdos_n30_p0.4.npy')
e3Y = np.load('synthetic_set/Y_syn_erdos_n30_p0.3.npy')
e2Y = np.load('synthetic_set/Y_syn_erdos_n30_p0.2.npy')
e1Y = np.load('synthetic_set/Y_syn_erdos_n30_p0.1.npy')


ptotaalerdosY = np.concatenate((e9Y,e8Y,e7Y,e6Y,e5Y,e4Y,e3Y,e2Y,e1Y), axis = 0 )
print(ptotaalerdosY.shape) #(990,30,31)
np.save('synthetic_set/Ytotalsynt_erdos_n30',ptotaalerdosY)

###################################################################################################################################################
#watts synthetic (Dataset 5.2)

# k =12 en p = 0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95 elk 100

p0515X = np.load('synthetic_set/X_syn_watts_n30_k12_p0515.npy')
p1525X = np.load('synthetic_set/X_syn_watts_n30_k12_p1525.npy')
p2535X = np.load('synthetic_set/X_syn_watts_n30_k12_p2535.npy')
p3545X = np.load('synthetic_set/X_syn_watts_n30_k12_p3545.npy')
p4555X = np.load('synthetic_set/X_syn_watts_n30_k12_p4555.npy')
p5565X = np.load('synthetic_set/X_syn_watts_n30_k12_p5565.npy')
p6575X = np.load('synthetic_set/X_syn_watts_n30_k12_p6575.npy')
p7585X = np.load('synthetic_set/X_syn_watts_n30_k12_p7585.npy')
p8595X = np.load('synthetic_set/X_syn_watts_n30_k12_p8595.npy')
p9505X = np.load('synthetic_set/X_syn_watts_n30_k12_p9505.npy')

ptotaalwattsX = np.concatenate((p0515X,p1525X,p2535X,p3545X,p4555X,p5565X,p6575X,p7585X,p8595X,p9505X), axis = 0)
print(ptotaalwattsX.shape) #(1000,30,30)
np.save('synthetic_set/Xtotalsynt_watss_n30', ptotaalwattsX)


p0515Y = np.load('synthetic_set/Y_syn_watts_n30_k12_p0515.npy')
p1525Y = np.load('synthetic_set/Y_syn_watts_n30_k12_p1525.npy')
p2535Y = np.load('synthetic_set/Y_syn_watts_n30_k12_p2535.npy')
p3545Y = np.load('synthetic_set/Y_syn_watts_n30_k12_p3545.npy')
p4555Y = np.load('synthetic_set/Y_syn_watts_n30_k12_p4555.npy')
p5565Y = np.load('synthetic_set/Y_syn_watts_n30_k12_p5565.npy')
p6575Y = np.load('synthetic_set/Y_syn_watts_n30_k12_p6575.npy')
p7585Y = np.load('synthetic_set/Y_syn_watts_n30_k12_p7585.npy')
p8595Y = np.load('synthetic_set/Y_syn_watts_n30_k12_p8595.npy')
p9505Y = np.load('synthetic_set/Y_syn_watts_n30_k12_p9505.npy')

ptotaalwattsY = np.concatenate((p0515Y,p1525Y,p2535Y,p3545Y,p4555Y,p5565Y,p6575Y,p7585Y,p8595Y,p9505Y), axis = 0)
print(ptotaalwattsY.shape) #(1000,30,31)
np.save('synthetic_set/Ytotalsynt_watss_n30', ptotaalwattsY)

###################################################################################################################################################
#barbasi synthetic (Dataset 5.3)

# n =30 en m = 2,3,4,5,6,7,8,9,(5x12),(5x17),(5x22),(6x27), elke is 35


m2529X = np.load('synthetic_set/X_syn_basi_n30_m2529.npy')
m2024X = np.load('synthetic_set/X_syn_basi_n30_m2024.npy')
m1519X = np.load('synthetic_set/X_syn_basi_n30_m1519.npy')
m1014X = np.load('synthetic_set/X_syn_basi_n30_m1014.npy')
m9X = np.load('synthetic_set/X_syn_basi_n30_m9.npy')
m8X = np.load('synthetic_set/X_syn_basi_n30_m8.npy')
m7X = np.load('synthetic_set/X_syn_basi_n30_m7.npy')
m6X = np.load('synthetic_set/X_syn_basi_n30_m6.npy')
m5X = np.load('synthetic_set/X_syn_basi_n30_m5.npy')
m4X = np.load('synthetic_set/X_syn_basi_n30_m4.npy')
m3X = np.load('synthetic_set/X_syn_basi_n30_m3.npy')
m1X = np.load('synthetic_set/X_syn_basi_n30_m1.npy')

mtotaalbasiX = np.concatenate((m1X,m3X,m4X,m5X,m6X,m7X,m8X,m9X,m1014X,m1519X,m2024X,m2529X), axis= 0)
# mtotaalbasiX = np.concatenate((m2529X,m2024X,m1519X,m1014X,m9X,m8X,m7X,m6X,m5X,m4X,m3X,m1X), axis = 0 )
print(mtotaalbasiX.shape) #(1015,30,30)
# np.save('synthetic_set/Xtotalsynt_basi_n30',mtotaalbasiX)


m2529Y = np.load('synthetic_set/Y_syn_basi_n30_m2529.npy')
m2024Y = np.load('synthetic_set/Y_syn_basi_n30_m2024.npy')
m1519Y = np.load('synthetic_set/Y_syn_basi_n30_m1519.npy')
m1014Y = np.load('synthetic_set/Y_syn_basi_n30_m1014.npy')
m9Y = np.load('synthetic_set/Y_syn_basi_n30_m9.npy')
m8Y = np.load('synthetic_set/Y_syn_basi_n30_m8.npy')
m7Y = np.load('synthetic_set/Y_syn_basi_n30_m7.npy')
m6Y = np.load('synthetic_set/Y_syn_basi_n30_m6.npy')
m5Y = np.load('synthetic_set/Y_syn_basi_n30_m5.npy')
m4Y = np.load('synthetic_set/Y_syn_basi_n30_m4.npy')
m3Y = np.load('synthetic_set/Y_syn_basi_n30_m3.npy')
m1Y = np.load('synthetic_set/Y_syn_basi_n30_m1.npy')

mtotaalbasiY = np.concatenate((m2529Y,m2024Y,m1519Y,m1014Y,m9Y,m8Y,m7Y,m6Y,m5Y,m4Y,m3Y,m1Y), axis = 0 )
print(mtotaalbasiY.shape) #(1015,30,31)
np.save('synthetic_set/Ytotalsynt_basi_n30',mtotaalbasiY)

###################################################################################################################################################
#total synthetic (Dataset 5)
x_basi = np.load('synthetic_set/Xtotalsynt_basi_n30.npy')
x_watts = np.load('synthetic_set/Xtotalsynt_watss_n30.npy')
x_erdos = np.load('synthetic_set/Xtotalsynt_erdos_n30.npy')

y_basi = np.load('synthetic_set/Ytotalsynt_basi_n30.npy')
y_watts = np.load('synthetic_set/Ytotalsynt_watss_n30.npy')
y_erdos = np.load('synthetic_set/Ytotalsynt_erdos_n30.npy')

x_synth_totaal = np.concatenate((x_basi,x_watts,x_erdos), axis = 0 )
y_synth_totaal = np.concatenate((y_basi,y_watts,y_erdos), axis = 0 )

np.save('synthetic_set/x_synth_totaal', x_synth_totaal)
np.save('synthetic_set/y_synth_totaal', y_synth_totaal)
print(x_synth_totaal.shape)
print(y_synth_totaal.shape)