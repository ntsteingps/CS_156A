import numpy as np 
from random import *
import matplotlib.pyplot as plt

def getdata(npoints):
	#first generate the target function f using two random points p1 and p2
	#Define target function f
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
	#here we're going to evaluate whether a given point should be signed + or - according to our target function f
	a = (point[0]-p1[0])*(p2[1]-p1[1])
	b = (point[1]-p1[1])*(p2[0]-p1[0])
	return np.sign(a-b)

def PLA2(points,niterations):
	#now let's do the PLA 
	#we input the points from getdata as well as the number of iterations we want to run our PLA

	#define our initial weight vector which is 0

	w = np.array((0.,0.,0.))

	X = np.empty((npoints,3),dtype=float)

	X[:,0] = 1.
	X[:,1:3] = points[:,0:2]

	count = 0 


	for i in range(nloop):
		#h = np.empty(npoints,dtype=int)
		h = []
		for j in range(npoints):
			h.append(np.sign(np.dot(w,X[j,:])))

		idxs = []
		
		for k in range(npoints):	
			if h[k] != points[k,2]:
				idxs.append(k)
		
		if len(idxs) == 1 or len(idxs) == 0:
			if w[0] == 0:
				print(len(idxs),idxs,w,X[idx,:],points[idx,2],h)
			break
		
		idx = choice(idxs)

		#print(i, w, points[idx,2],X[idx,:])

		w = w + points[idx,2]*X[idx,:]
		
		count = count + 1

	return w, count 

def countmisclassified(nrandompoints,f,w):
	ncount = 0.
	for i in range(nrandompoints):
		randx = uniform(-1.,1.)
		randy = uniform(-1.,1.)
		feval = evaluatef((0.1,f[0]*0.1+f[1]),(0.2,f[0]*0.2+f[1]),(randx,randy))
		gpoint = (1,randx,randy)
		geval = np.sign(np.dot(w,gpoint))
		if feval != geval:
			ncount = ncount + 1.
	return ncount 


npoints = 100
niterations = 100
nloop = 1000
nrandompoints = 1000

countavg = []
probavg = []

for i in range(niterations):
	points, f = getdata(npoints)
	w, count = PLA2(points,niterations)
	countavg.append(count)
	slope = -(w[0]/w[2])/(w[0]/w[1])
	intercept = -w[0]/w[2]
	ncount = countmisclassified(nrandompoints,f,w)
	probavg.append(ncount/float(nrandompoints))

	#print probavg

	for i in range(len(points)):
		if points [i,2] == 1:
			plt.plot(points[i,0],points[i,1],'go')
		if points[i,2] == -1:
			plt.plot(points[i,0],points[i,1],'ro')

	xarray = np.linspace(-1.,1.,10)

	plt.plot(xarray,f[0]*xarray+f[1],'k--')
	plt.plot(xarray,slope*xarray+intercept,'b--')
	plt.xlim(-1.,1.)
	plt.ylim(-1.,1.)
	#plt.show()


print(np.average(countavg))
print(np.average(probavg))



