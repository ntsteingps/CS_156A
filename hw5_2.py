import numpy as np
from random import *

N = 100 #Number of training points 
eta = 0.01 #Learning rate 
Nexp = 100 #Number of experiments 
thresh = 0.01 #Threshold we must be less than to stop 
nloops = 100 #Set the max number of loops
weights = np.array([0.0,0.0,0.0]) #set the initial weight vector 
ntest = 1000 #set a large number of test points to calculate Eout

def getdata(npoints):
	#first generate the target function f using two random points p1 and p2 in our space X
	#Define target function f (f[0]=slope, f[1]=intercept)
	p1 = np.array((uniform(-1.,1.),uniform(-1.,1.)))
	p2 = np.array((uniform(-1.,1.),uniform(-1.,1.)))
	f = np.empty(2,dtype=float)
	f[0] = (p2[1]-p1[1])/(p2[0]-p1[0]) #slope
	f[1] = p1[1] - f[0]*p1[0] #intercept
	#now let's get our npoints [location,f(x)] or [x,y,f(x)]
	points = np.empty((npoints,3),dtype=float)
	for i in range(npoints):
		points[i,0] = uniform(-1.,1.)
		points[i,1] = uniform(-1.,1.)
		points[i,2] = evaluatef(p1,p2,points[i,0:2])
	return points, f

def evaluatef(p1,p2,point):
	#here we're going to evaluate whether a given point should be signed + or - according to our target function f defined by p1 and p2
	slope = (p2[1]-p1[1])/(p2[0]-p1[0]) #slope
	intercept = p1[1] - slope*p1[0] #intercept
	p1 = (0.1,slope*0.1+intercept)	
	p2 = (0.2,slope*0.2+intercept)
	a = (point[0]-p1[0])*(p2[1]-p1[1])
	b = (point[1]-p1[1])*(p2[0]-p1[0])
	return np.sign(a-b)

def LRwSGD(points, eta, w, thresh):
	#This is a function to perform the Logistic Regression with Stochastic Gradient Descent 
	#Our inputs are the generated points, the learning rate, the initial weight vector, and the threshold 

		w = [0, 0, 0]
		X = np.empty((N, 4), dtype=float)
		X[:, 0] = 1.
		X[:, 1:4] = points[:, 0:3]
		diff = 100 # set initial value to be high
		epoch = 0

		while diff > thresh:
			epoch = epoch + 1
			w0 = np.copy(w)
			shuffle(X)
			for i in range(len(X[:,0])):
				Ein = SGD(X[i,:],w0)
				w0 = w0 - (eta*Ein)
			w_t_1 = w0
			diff = w_t_1 - w
			diffsquare = 0.
			for i in range(len(diff)):
				diffsquare = diffsquare + (diff[i])**2.
			diff = diffsquare**(1./2.)
			w = w_t_1

		return w, epoch

def SGD(point, w):
    return -1.*(np.array(point[:3]) * point[3])/(1.0 + np.exp(point[3]*np.dot(w, point[:3])))

def countmisclassified(nrandompoints,f,w):
	#this is a function I made to calculate the probability that f and g will disagree on their classification of a random point
	#we're calculating/returning the number of misclassified points from a large number of random points 
	ncount = 0.
	for i in range(nrandompoints):
		randx = uniform(-1.,1.)
		randy = uniform(-1.,1.)
		feval = evaluatef((0.1,f[0]*0.1+f[1]),(0.2,f[0]*0.2+f[1]),(randx,randy))
		gpoint = (1,randx,randy)
		geval = np.sign(np.dot(np.transpose(w),gpoint))
		if feval != geval:
			ncount = ncount + 1.
	return ncount 

epochs = []
Eout = []

for i in range(Nexp):
	print(i)
	points, f = getdata(N)
	p1 = (0.1,f[0]*0.1+f[1])	
	p2 = (0.2,f[0]*0.2+f[1])
	w, epoch = LRwSGD(points, eta, weights, thresh)
	epochs.append(epoch)
	ncount = countmisclassified(ntest,f,w)
	Eout.append(ncount/ntest)
#	if (ncount/ntest) < .2:
#		Eout.append(ncount/ntest)
#	else:
#		Eout.append(1. - (ncount/ntest))

#print Eout 
print(np.average(Eout))
print(np.average(epochs))
