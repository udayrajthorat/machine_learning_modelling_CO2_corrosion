# program to calculate rbf response surface given DoE points and responses

import timeit
start = timeit.default_timer()


from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import pandas as pd
from sklearn.preprocessing import scale # To scale the data
from sklearn import decomposition

# if the file is in .xlsx

# with open('All_values_CR_velocity.txt', 'w') as file:
#     pd.read_excel('All_values_CR_velocity.xlsx').to_string(file, index=False,header=None)


# reading data into a file
xydata = open('All_values_CR_velocity.txt','r').readlines()

x1 = []
x2 = []
x3 = []
x4 = []
corrosion_rate= []

for line in xydata:
    x1.append(float(line.split()[0]))
    x2.append(float(line.split()[1]))
    x3.append(float(line.split()[2]))
    x4.append(float(line.split()[3]))
    corrosion_rate.append(float(line.split()[4]))
    
print ('read finished')


# dimensional DoE points
x_dim = (np.vstack([x1,x2,x3,x4])).transpose()

# Calculating number of DoE points for matrix x
n=int(x_dim.shape[0])

# Calculating number of dvs
ndv=int(x_dim.shape[1])
print ('n = {0:d} ndv = {1:d}'.format(n,ndv))

# scale x1 and x2 design variables to lie between 0 and 1
x1_scaled = []
x2_scaled = []
x3_scaled = []
x4_scaled = []

for i in range(0,n):
    x1_scaled.append((x1[i]-min(x1))/(max(x1)-min(x1)))
    x2_scaled.append((x2[i]-min(x2))/(max(x2)-min(x2)))
    x3_scaled.append((x3[i]-min(x3))/(max(x3)-min(x3)))
    x4_scaled.append((x4[i]-min(x4))/(max(x4)-min(x4)))
    

# vector of scaled DoE points
x_nondim = (np.vstack([x1_scaled,x2_scaled,x3_scaled,x4_scaled])).transpose()

# array of floats needed for use in ANN functions
x_scaled = np.array(x_nondim,dtype='float64')

# Now scale the objectives to lie between 0 and 1: objective 1 =
minobj1 = min(corrosion_rate)
denom1 = (max(corrosion_rate)-min(corrosion_rate))

o1 = []
for i in range(0,len(corrosion_rate)):
    o1.append((corrosion_rate[i]-minobj1)/denom1)
    
    
### Split the data ### 

from sklearn.model_selection import train_test_split
x_train, x_test, o1_train, o1_test = train_test_split(x_scaled, o1, test_size = 0.30,random_state=42)


# x_train = x_scaled[20:]
# x_test = x_scaled[:20]

# o1_train = o1[20:]
# o1_test = o1[:20]


# array of floats needed for use in ANN functions
obj = np.array(o1_train,dtype='float64')

# Surrogate modelling of objective 1 - corrosion rates - using ANNs

# Sequential class - exactly creates ANN initialized as sequence of layers.structure.
ann_tr = tf.keras.models.Sequential()


units = 15

batch_size  = 5

epochs = 10000

import os
import random

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)
tf.random.set_seed(1234)


# Adding the input layer and the hidden layer

initializers = {
  "kernel_initializer": tf.keras.initializers.RandomNormal(seed=1337),
  "bias_initializer": tf.keras.initializers.Constant(value=0.1)    
}

# Use of Dense class - connection between input layer and hidden layer
ann_tr.add(tf.keras.layers.Dense(units = units, activation ='relu', **initializers))

# Use of Dense class - connection between input layer and hidden layer
ann_tr.add(tf.keras.layers.Dense(units = units, activation ='relu', **initializers))

# Use of Dense class - connection between input layer and hidden layer
ann_tr.add(tf.keras.layers.Dense(units = units, activation ='relu', **initializers))

# Use of Dense class - connection between input layer and hidden layer
ann_tr.add(tf.keras.layers.Dense(units = units, activation ='relu', **initializers))

# Use of Dense class - connection between input layer and hidden layer
# ann_tr.add(tf.keras.layers.Dense(units = units, activation ='relu', **initializers))


# Adding the output layer
ann_tr.add(tf.keras.layers.Dense(units=1))

# Compiling the ANN - the tool with which we perform stochastic Gradient descent.    
ann_tr.compile(optimizer = 'adam', loss = "mean_squared_error" )

# Training the ANN model on the whole data set - no data partitioning
ann_tr.fit(x_train, obj, batch_size= batch_size, epochs = epochs)


obj_pred = ann_tr.predict(x_test)


# Single objective optimisation of objective - corrosion rate - using ANNs
from scipy import optimize

def obj1_f(x): # function to be optimised
    xdash = np.zeros((1,4),dtype='float64') # need compatible data for optimizer
    xdash[0][0] = x[0]
    xdash[0][1] = x[1]
    xdash[0][2] = x[2]
    xdash[0][3] = x[3]
    return ann_tr.predict(xdash)

# testing obj1_f for data input compatibility
x = [0.5,0.5,0.5,0.5]
yval = obj1_f(x)

# set optimisation bounds 
bnds = ((0,1),(0,1),(0,1),(0,1)) 

optmethod = "Nelder-Mead"   # set optimisation method

res = optimize.minimize(obj1_f,x, bounds=bnds, method="Nelder-Mead",options={'xatol': 1e-8, 'disp': True})


print("res success {0:5d}".format(res.success))
print("optimal scaled points {0:10.5e} {1:10.5e} {2:10.5e} {3:10.5e}"\
      .format(res.x[0], res.x[1], res.x[2], res.x[3]))
    
print("optimal value {0:10.5e}".format(res.fun))

xopt = res.x[0]
yopt = res.x[1]
optval = res.fun

#### Create 2X2 surrogate ####

# -----------------------------------------------------
# plot for two design variables
# -----------------------------------------------------

# # create output points for displaying the surface
na = 41
xa = np.zeros((na*na,ndv))
ya_exact = np.zeros((na*na))

deltax = 1.0/(na-1)
deltay = deltax
countpos = 0
for i in range(0,na):
    xval = i*deltax
    for j in range(0,na):
        yval = j*deltay
        xa[countpos][1] = xval
        xa[countpos][3] = yval
        countpos = countpos + 1

ya_ann_cr = ann_tr.predict(xa)

Xr = np.zeros((41,41))
Yr = np.zeros((41,41))
Zr_ann_cr = np.zeros((41,41));
ipos = -1;
for i in range(0,41):
    for j in range(0,41):
        ipos = ipos + 1
        Xr[i][j]=xa[ipos][1]        
        Yr[i][j]=xa[ipos][3]               
        Zr_ann_cr[i][j]=ya_ann_cr[ipos]
        


fig = plt.figure(figsize=(10,8))
fig = plt.gcf()
fig.suptitle('ANN approximation of CR- T_u',fontsize=20)
ax = fig.gca(projection='3d')
ax=plt.axes(projection='3d')
surf = ax.plot_surface(Xr, Yr, Zr_ann_cr, rstride=1, cstride=1, alpha=0.9, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_zlim(0, 1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# ax.set_zticks(np.arange(0, 0.51, 0.1))
ax.tick_params(axis='z', labelsize=10)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)


# plot out the scatter points

#ax.scatter(x_scaled[:,0],x_scaled[:,1],o1, c='r', marker='o',s=8)
# ax.scatter(xopt,yopt,optval, c='k', marker='*',s=30)   # plot optimum point

ax.set_xlabel('pCO2')
ax.set_ylabel('u')
ax.set_zlabel('Corrosion rate')

# Set the color bar
fig = plt.figure(figsize=(10,8))
mappable = ax.collections[0]
cb = plt.colorbar(mappable, shrink=1.0, aspect=6)
cb.set_label('Corrosion rate', fontsize=14)

fig.savefig('ANN_CR_pCO2_u.jpg')



########################### Optimal values ####################################


# x1opt= res.x[0]*(max(x1)-min(x1))+min(x1)
# x2opt = res.x[1]*(max(x2)-min(x2))+min(x2)
# x3opt = res.x[2]*(max(x3)-min(x3))+min(x3)
# x4opt = res.x[3]*(max(x4)-min(x4))+min(x4)

# print("Optimal values pH = {0:10.1e} pCO2 = {1:10.5e} T = {2:10.5e} u = {3:10.5e}".format(x1opt,x2opt,x3opt,x4opt))


# values = {"Parameters": ["pH","pCO2 (bar)","T (C)","u (m/s)"], "Values":[x1opt,x2opt,x3opt,x4opt]}

# df = pd.DataFrame(values)

# df.to_excel("Optimal values NM ANN.xlsx")

############################### Metrics values ################################

# from sklearn import metrics

# MSE = metrics.mean_squared_error(o1_test,obj_pred)

# RMSE = np.sqrt(MSE)



# print("Mean Squared Error = {0:2}".format (MSE))   
  
# print("Root Mean Square Error = {0:2}".format (RMSE))   

# from sklearn.metrics import r2_score
# R2_ANN = r2_score(o1_test,obj_pred)

# print("R-sqaure = {0:2}".format (R2_ANN))   




# fig = plt.figure()
# fig = plt.gcf()

# fig.set_size_inches(7,5)

# ax = fig.add_axes([0,0,1,1])

# #plt.rc('grid', linestyle="-", color='black')

# plt.plot(o1,o1,color='black')

# plt.scatter(o1_test,obj_pred,marker="o",color='black')


# plt.xlim(0,1.01)
# plt.ylim(0,1.01)

# #plt.legend (['Current model','Empirical correlation','Nesic et al.(1995)'], loc='upper left')

# plt.xlabel ("Actual corrosion rate (mm/yr)")
# plt.ylabel ("Predicted corrosion rate (mm/yr)")

# plt.grid(linestyle=':')
    
# plt.savefig('ANN-Predictions',bbox_inches='tight', dpi=150)

# stop = timeit.default_timer()
# execution_time = stop - start

# print("Program Executed in "+str(execution_time)) # It returns time in seconds