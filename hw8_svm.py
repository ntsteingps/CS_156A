import random
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

#Problem 11 (solved w/ both geometric and matrix forms)
x = np.array([[1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2,0]])
y = np.array([-1, -1, -1, 1, 1, 1, 1])
z = []

for i in range(len(x)):
    z_temp = []
    z_temp.append(x[i,1]**2 - 2*x[i,0] - 1)
    z_temp.append(x[i,0]**2 - 2*x[i,1] + 1)
    z.append(z_temp)

lfit = svm.SVC(kernel = 'linear', C = 1e6)
lfit.fit(z,y)
ytemp = lfit.predict(z)
print('w = ', lfit.coef_)
print('b = ', lfit.intercept_)

a = np.empty(len(y),dtype=float)
w_options = np.array(([-0.5,-1.,1.],[-0.5,1.,-1.],[-0.5,1.,0.],[-0.5,0.,1.]))
fillmat = np.ones((len(x),3))
fillmat[:,1:3] = z
for w in w_options:
	b = np.dot(fillmat,np.transpose(w))
	a = np.vstack((a,b))

distance = np.sum(np.multiply(y,a),axis=1)

print(distance)

#Problem 12
lfit = svm.SVC(kernel = 'poly', C = 1e6, degree = 2, gamma = 1, coef0 = 1)
lfit.fit(x,y)
ytemp = lfit.predict(x)
print('number of support vectors = ', lfit.support_vectors_.shape[0])
