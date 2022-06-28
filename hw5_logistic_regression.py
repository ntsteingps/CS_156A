import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import matrix
from numpy import linalg

#Written by N. Stein 11/3/18
#Logistic Regression Algorithm for HW 5

def evaluate_sign(v1, v2):
    # if the dot product of x and w is positive, y = +1, if negative, y = -1
    return 1 if np.dot(v1, v2) > 0 else -1

def evaluate_sign_initial(v1,v2):
    # find two points on the line w and compare with the input point to see what side it is on
    x_w_1 = 0.25
    x_w_2 = 0.5
    y_w_1 = v1[1]*x_w_1 + v1[2]
    y_w_2 = v1[1]*x_w_2 + v1[2]
    return 1 if (v2[1] - x_w_1)*(y_w_2 - y_w_1) - (v2[2]-y_w_1)*(x_w_2-x_w_1) > 0 else -1

def check_classification(data, w):
    # check to see if points are misclassified
    # takes in data points and weight vector
    misclassified_points = []
    for i in range(len(data)):
        x = data[i][0:3]
        y = data[i][3]
        if evaluate_sign(w, x) != y:
            # then the point is misclassified
            misclassified_points.append(x)
    return misclassified_points

def gradient_E_in(num_points,data,w):
    summand = 0
    order = [k for k in range(num_points)]
    random.shuffle(order) # create a random ordering of indices to sample during each epoch
    for i in range(num_points):
        x = data[order[i]][0:3]
        y = data[order[i]][3]
        prod_xy = [y*j for j in x]
        summand += np.divide(prod_xy, 1+np.exp(y*np.dot(np.transpose(w), x)))
    return -1*summand/num_points


# generate the target function and a set of training points
def generate_data(num_points):
    # generate target function f (w) by choosing a random line in the target function
    # we will take two random, uniformly distributed points in [-1,1]x[-1,1] and take the line passing through them
    point1 = [random.uniform(-1, 1), random.uniform(-1, 1)]
    point2 = [random.uniform(-1, 1), random.uniform(-1, 1)]
    f = [0,0,0]
    f[1] = (point2[1]-point1[1])/(point2[0]-point1[0])  # slope
    f[2] = point1[1] - f[1]*point1[0]  # y int

    # choose the inputs x_n of the data set as random points uniformly in x
    data_points = []
    for j in range(num_points):
        pt = [1,random.uniform(-1, 1), random.uniform(-1, 1), 1]  # randomly generated x1 and x2 and choose y = 1
        pt[3] = evaluate_sign_initial(f, pt)  # evaluate point to select the correct y value for the target function f
        data_points.append(pt)

    return data_points, f

# Start the PLA with the weight vector w being all zeros

def main():
    n_points = 100  # number of training points | set up with n_points = 10 for q8
    condition = 0.01 # stop the algorithm when ||w^(t-1) - w^(t) < 0.01
    n_points_Eout = 5000
    n_iterations = 100
    eta = 0.01 # learning rate

    n_misclassified_out_total = 0
    count = 0
    for q in range(n_iterations):
        print(q)
        data, f = generate_data(n_points)
        w_diff = 1
        w = [0, 0, 0]
        while w_diff > condition:
            w_prev = list(np.copy(w))
            order = [k for k in range(n_points)]
            random.shuffle(order)  # create a random ordering of indices to sample during each epoch
            for i in range(n_points):
                x = data[order[i]][0:3]
                y = data[order[i]][3]
                prod_xy = [y * j for j in x]
                grad = -1*np.divide(prod_xy, 1 + np.exp(y * np.dot(np.transpose(w_prev), x)))
                w_prev -= eta*grad  # update the weights
            w_t_1 = w_prev
            w_diff = np.linalg.norm(w - w_t_1)
            w     = w_t_1
            count = count + 1

        # Check E out
        # generate n_points_Eout data points and check how well they are classified
        data_Eout = []
        for j in range(n_points_Eout):
            pt = [1,random.uniform(-1, 1), random.uniform(-1, 1), 1]  # randomly generated x1 and x2 and choose y = 1
            pt[3] = evaluate_sign_initial(f, pt)  # evaluate point to select the correct y value for the target function f
            data_Eout.append(pt)
        misclassified_out = check_classification(data_Eout, w)
        n_misclassified_out = len(misclassified_out)
        n_misclassified_out_total += n_misclassified_out

    return f, w, n_misclassified_out_total, n_iterations, n_points_Eout, count

f, w, n_misclassified_out, n_iterations, n_points_Eout, count = main()
print(f, w, n_misclassified_out/n_iterations/n_points_Eout, count/n_iterations)
