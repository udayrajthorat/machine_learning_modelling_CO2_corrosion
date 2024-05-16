# program to calculate SVR response surface given DoE points and responses
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import timeit
start = timeit.default_timer()

import matplotlib.pyplot as plt
import numpy as np
import math 

import pandas as pd
from sklearn.preprocessing import scale # To scale the data
from sklearn import decomposition


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
    
### Split the data ### 

from sklearn.model_selection import train_test_split
x_train, x_test, o1_train, o1_test = train_test_split(x_scaled, o1, test_size = 0.3, random_state = 0)  


# array of floats needed for use in ANN functions
obj1 = np.array(o1_train,dtype='float64')

# Surrogate modelling of objective using Random Forests
from sklearn.ensemble import RandomForestRegressor 

# create regressor object 
rf_regressor = RandomForestRegressor(n_estimators = 10, random_state = 42, bootstrap=True,\
                                     max_depth=10)

rf_cr = rf_regressor.fit(x_train, obj1)

obj_pred = rf_cr.predict(x_test)

    
# ----------------------------------------------------------------------------------------
# Single objective optimisation of corrosion using Random Forests
# ----------------------------------------------------------------------------------------

# Single objective optimisation of objective - corrosion rate - using ANNs
from scipy import optimize

def obj1_f(x): # function to be optimised
    xdash = np.zeros((1,4),dtype='float64') # need compatible data for optimizer
    xdash[0][0] = x[0]
    xdash[0][1] = x[1]
    xdash[0][2] = x[2]
    xdash[0][3] = x[3]
    return rf_cr.predict(xdash)

# testing obj1_f for data input compatibility
x = [0.5,0.5,0.5,0.5]
yval = obj1_f(x)

# set optimisation bounds 
bnds = ((0,1),(0,1),(0,1),(0,1)) 

optmethod = "Powell" #"Nelder-Mead"   # set optimisation method

# optimise with scipy functions
if (optmethod=="Nelder-Mead"):
    res = optimize.minimize(obj1_f,x, bounds=bnds, method="Nelder-Mead")
elif (optmethod=="Powell"):
     res = optimize.minimize(obj1_f,x, bounds=bnds, method="Powell")

print("res success {0:5d}".format(res.success))
print("optimal scaled points {0:10.5e} {1:10.5e} {2:10.5e} {3:10.5e}"\
      .format(res.x[0], res.x[1], res.x[2], res.x[3]))
    
print("optimal value {0:10.5e}".format(res.fun))
xopt = res.x[0]
yopt = res.x[1]
optval = res.fun

# calculate RMSE for cross validation
from sklearn import metrics

MSE = metrics.mean_squared_error(o1_test,obj_pred)

RMSE = np.sqrt(MSE)

print("Mean Squared Error = {0:2}".format (MSE))   
print("Root Mean Square Error = {0:2}".format (RMSE))   


from sklearn.metrics import r2_score
R2 = r2_score(o1_test,obj_pred)

print("R-sqaure = {0:2}".format (R2))



fig = plt.figure()
fig = plt.gcf()

fig.set_size_inches(7,5)

ax = fig.add_axes([0,0,1,1])

#plt.rc('grid', linestyle="-", color='black')

plt.plot(o1,o1,color='black')

plt.scatter(o1_test,obj_pred,marker="o",color='black')


plt.xlim(0,1.01)
plt.ylim(0,1.01)

#plt.legend (['Current model','Empirical correlation','Nesic et al.(1995)'], loc='upper left')

plt.xlabel ("Actual corrosion rate (mm/yr)")
plt.ylabel ("Predicted corrosion rate (mm/yr)")

plt.grid(linestyle=':')
    
plt.savefig('RF-Predictions',bbox_inches='tight', dpi=150)

plt.show()



stop = timeit.default_timer()
execution_time = stop - start

print("Program Executed in "+str(execution_time)) # It returns time in seconds


# Features ranking needs to be included to understand which parameter contributed to make the decision

# -----------------------------------------------------
# plot for two design variables
# -----------------------------------------------------

# # create output points for displaying the surface
# na = 41
# xa = np.zeros((na*na,ndv))
# ya_exact = np.zeros((na*na))

# deltax = 1.0/(na-1)
# deltay = deltax
# countpos = 0
# for i in range(0,na):
#     xval = i*deltax
#     for j in range(0,na):
#         yval = j*deltay
#         xa[countpos][0] = xval
#         xa[countpos][1] = yval
#         countpos = countpos + 1

# # create outputs for displaying the Random Forest surfaces
# ya_rf_cr = rf_cr.predict(xa)

# Xr = np.zeros((41,41))
# Yr = np.zeros((41,41))
# Zr_rf_cr = np.zeros((41,41));
# ipos = -1;
# for i in range(0,41):
#     for j in range(0,41):
#         ipos = ipos + 1
#         Xr[i][j]=xa[ipos][0]        
#         Yr[i][j]=xa[ipos][1]               
#         Zr_rf_cr[i][j]=ya_rf_cr[ipos]
        
# fig = plt.figure(1)
# fig.suptitle('Random Forest approximation of CR',fontsize=10)
# #ax = fig.gca(projection='3d')
# ax=plt.axes(projection='3d')
# surf = ax.plot_surface(Xr, Yr, Zr_rf_cr, rstride=8, cstride=8, alpha=0.3, cmap=cm.coolwarm,
#         linewidth=0, antialiased=False)
# ax.set_zlim(-0.1, 1.1)
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# #fig.colorbar(surf, shrink=0.5, aspect=5)

# # plot out the scatter points
# ax.scatter(x_scaled[:,0],x_scaled[:,1],obj1, c='r', marker='o',s=8)
# ax.scatter(xopt,yopt,optval, c='k', marker='o',s=16)   # plot optimum point
# ax.set_ylabel('x2')
# ax.set_zlabel('Objective 1')
# fig.savefig('RF_CR.jpg')

 
