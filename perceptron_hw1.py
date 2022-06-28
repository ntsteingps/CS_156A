import random
import numpy as np
import matplotlib.pyplot as plt

def evaluate_sign(v1, v2):
    # if the dot product of x and w is positive, y = +1, if negative, y = -1
    #return 1 if np.dot(v1, v2) > 0 else -1
    #find two points on the line w and compare with the input point to see what side it is on
    x_w_1 = 0.25
    x_w_2 = 0.5
    y_w_1 = v1[0]*x_w_1 + v1[1]
    y_w_2 = v1[0]*x_w_2 + v1[1]
    return 1 if (v2[0] - x_w_1)*(y_w_2 - y_w_1) - (v2[1]-y_w_1)*(x_w_2-x_w_1) > 0 else -1

def check_classification(data, w):
    # check to see if points are misclassified
    # takes in data points and weight vector
    misclassified_points = []
    for i in range(len(data)):
        x = data[i][:]
        y = data[i][2]
        if evaluate_sign(w, x) != y:
            # then the point is misclassified
            misclassified_points.append(x)
    return misclassified_points


def generate_data(num_points):
    # generate target function f (w) by choosing a random line in the target function
    # we will take two random, uniformly distributed points in [-1,1]x[-1,1] and take the line passing through them
    point1 = [random.uniform(-1, 1), random.uniform(-1, 1)]
    point2 = [random.uniform(-1, 1), random.uniform(-1, 1)]
    f = [0,0,0]
    f[0] = (point2[1]-point1[1])/(point2[0]-point1[0])  # slope
    f[1] = point1[1] - f[1]*point1[0]  # y int
    #print(f)

    # choose the inputs x_n of the data set as random points uniformly in x
    data_points = []
    for j in range(num_points):
        pt = [random.uniform(-1, 1), random.uniform(-1, 1), 1]  # randomly generated x1 and x2 and choose y = 1
        pt[2] = evaluate_sign(f,pt)  # evaluate point to select the correct y value for the target function f
        data_points.append(pt)

    return data_points, f

# Start the PLA with the weight vector w being all zeros

def main():
    w_initial = [0, 0, 0]
    w_final   = w_initial
    n_points = 10  # number of training points
    n_iterations = 100  # number of iterations
    data, f = generate_data(n_points)
    for i in range(n_iterations):
        if i == 0:
            misclassified_points = check_classification(data, w_initial)

        # choose a random point out of the set of misclassified points and use it to update w
        if(misclassified_points):  # only do this if there are misclassified points
            random_wrong_point = random.choice(misclassified_points)
            w_final[0] = w_final[0] + random_wrong_point[0]*random_wrong_point[2]
            w_final[1] = w_final[1] + random_wrong_point[1]*random_wrong_point[2]
            misclassified_points = check_classification(data, w_final)

        #print(w_final, len(misclassified_points),i)
        if len(misclassified_points) == 0:
            linarray = np.linspace(-1,1,100)
            plt.plot(linarray,w_final[0]*linarray + w_final[1],'r--')
            plt.xlim(-1,1)
            plt.ylim(-1,1)
            print(i, f, w_final)
            return i, data, f

    #print(w_final)

#average results over many runs
index = []
n_runs = 1
for q in range(n_runs):
    index_val, data, f = main()
    index.append(index_val)
    for i in range(len(data)):
        print(i)
        if data[i][2] == -1:
            plt.plot(data[i][0], data[i][1], 'go')
        else:
            plt.plot(data[i][0], data[i][1], 'ro')
    #print(index_val)
#average_index = sum(index) / float(len(index))
#print(average_index)

#plot the target function with the points
linarray = np.linspace(-1,1,100)
plt.plot(linarray,f[0]*linarray + f[1],'b--')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()

