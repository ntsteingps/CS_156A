import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import os

files = "test.csv"
os.chdir("C:/Users/ntste/Documents/Mars/Clay Dips/")
data = np.genfromtxt(files, delimiter=",", comments="#",dtype=float)

# regular grid covering the domain of the data
X,Y = np.meshgrid(np.arange(-60.0, 100.0, .1), np.arange(-60.0, 100.0, .1))
XX = X.flatten()
YY = Y.flatten()

order = 1    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    
    # evaluate it on grid
    Z = C[0]*X + C[1]*Y + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    
    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

# plot points and fitted surface
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
#ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
#plt.xlabel('X')
#plt.ylabel('Y')
#ax.set_zlabel('Z')
#ax.axis('equal')
#ax.axis('tight')
#plt.show()

v1 = [-1.*C[0], -1.*C[1]]
v2 = [1.,0.]

v3 = np.dot(v1,v2)

val = np.dot(np.linalg.norm(v1),np.linalg.norm(v2))

print(np.degrees(np.arccos(v3/val)))

#NEW - NATHAN CERTIFIES THIS IS CORRECT 2/6/2019. The above number is WRONG. USE SITE FRAME.
v1 = [-1.*C[0], -1.*C[1]]
print(C[0])
print(C[1])
v2 = [0.,1.]

v3 = np.dot(v1,v2)
val = np.dot(np.linalg.norm(v1),np.linalg.norm(v2))
#Make it wrap to 360 degrees
if C[0] > 0:
    dip = 360 - np.degrees(np.arccos(v3/val))
else:
    dip = np.degrees(np.arccos(v3/val))

#Flip dip by 180 degrees
if dip < 180:
    dip = dip + 180
else:
    dip = dip - 180
print(dip)
######

v1 = [C[0],C[1],1.]
v2 = [0.,0.,1.]

v3 = np.dot(v1,v2)

val = np.dot(np.linalg.norm(v1),np.linalg.norm(v2))
print(np.degrees(np.arccos(v3/val)))
