# program to calculate SVR response surface given DoE points and responses
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter

import timeit
start = timeit.default_timer()

import matplotlib.pyplot as plt
import tensorflow as tf

import pandas as pd
from sklearn.preprocessing import scale # To scale the data
from sklearn import decomposition
import numpy as np
import math

from sklearn.model_selection import GridSearchCV

#reading data into a file
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
    



# Surrogate modelling of objective- using SVR
from sklearn.svm import SVR


### Split the data ### 

from sklearn.model_selection import train_test_split
x_train, x_test, o1_train, o1_test = train_test_split(x_scaled, o1, test_size = 0.20, random_state = 20) 


# #############################################################################
# Fit regression model - choose only one at a time here
svr_rbf = SVR(kernel='rbf', C= 10, gamma=0.1, epsilon=0.1)

# {'C': 10, 'epsilon': 0.1, 'gamma': 0.1, 'kernel': 'rbf'}

svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
# #############################################################################
# examine accuracy of the various regressors
svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ['RBF', 'Linear', 'Polynomial']
model_color = ['m', 'c', 'g']

###############################################################################

svr_rbf_cr = svr_rbf.fit(x_train, o1_train)
R2_svr_rbf_cr = svr_rbf.score(x_train, o1_train)
print ("rbf R2 = {0:4.2e}".format(R2_svr_rbf_cr))
svr_lin_cr = svr_lin.fit(x_train, o1_train)
R2_svr_lin_cr = svr_lin.score(x_train, o1_train)
print ("lin R2 = {0:4.2e}".format(R2_svr_lin_cr))
svr_poly_cr = svr_poly.fit(x_train, o1_train)
R2_svr_poly_cr = svr_poly_cr.score(x_train, o1_train)
print ("poly R2 = {0:4.2e}".format(R2_svr_poly_cr))


obj_pred_rbf = svr_rbf_cr.predict(x_test)

obj_pred_lin = svr_lin_cr.predict(x_test)

obj_pred_poly = svr_poly_cr.predict(x_test)



# Single objective optimisation of objective - using SVR
from scipy import optimize

def obj1_f(x): # function to be optimised
    xdash = np.zeros((1,4),dtype='float64') # need compatible data for optimizer
    xdash[0][0] = x[0]
    xdash[0][1] = x[1]
    xdash[0][2] = x[2]
    xdash[0][3] = x[3]
    return abs(svr_poly_cr.predict(xdash))

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

x1opt= res.x[0]*(max(x1)-min(x1))+min(x1)
x2opt = res.x[1]*(max(x2)-min(x2))+min(x2)
x3opt = res.x[2]*(max(x3)-min(x3))+min(x3)
x4opt = res.x[3]*(max(x4)-min(x4))+min(x4)

print("Optimal values pH = {0:10.1e} pCO2 = {1:10.5e} T = {2:10.5e} u = {3:10.5e}".format(x1opt,x2opt,x3opt,x4opt))


values = {"Parameters": ["pH","pCO2 (bar)","T (C)","u (m/s)"], "Values":[x1opt,x2opt,x3opt,x4opt]}

df = pd.DataFrame(values)

df.to_excel("Optimal values NM SVR.xlsx")



# calculate RMSE for cross validation
MSE = np.square(np.subtract(o1_test,obj_pred_rbf)).mean()   

RMSE = math.sqrt(MSE)  


from sklearn.metrics import r2_score
R2 = r2_score(o1_test,obj_pred_rbf)

print("Mean Square Error rbf kernel = {0:2}".format (MSE)) 

print("Root Mean Square Error rbf kernel = {0:2}".format (RMSE)) 

print("R-sqaure = {0:2}".format (R2))


# MSE = np.square(np.subtract(o1_test,obj_pred_lin)).mean()   
# RMSE = math.sqrt(MSE)  

# print("Root Mean Square Error lin kernel = {0:2}".format (RMSE)) 

# MSE = np.square(np.subtract(o1_test,obj_pred_poly)).mean()   
# RMSE = math.sqrt(MSE)
# print("Root Mean Square Error poly kernel = {0:2}".format (RMSE)) 




fig = plt.figure()
fig = plt.gcf()

fig.set_size_inches(7,5)

ax = fig.add_axes([0,0,1,1])

#plt.rc('grid', linestyle="-", color='black')

plt.plot(o1,o1,color='black')

plt.scatter(o1_test,obj_pred_rbf,marker="o",color='black')


plt.xlim(0,1.01)
plt.ylim(0,1.01)

#plt.legend (['Current model','Empirical correlation','Nesic et al.(1995)'], loc='upper left')

plt.xlabel ("Actual corrosion rate (mm/yr)")
plt.ylabel ("Predicted corrosion rate (mm/yr)")

plt.grid(linestyle=':')
    
plt.savefig('SVR-Predictions',bbox_inches='tight', dpi=150)


stop = timeit.default_timer()
execution_time = stop - start

print("Program Executed in "+str(execution_time)) # It returns time in seconds






