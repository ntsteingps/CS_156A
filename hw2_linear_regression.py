import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import matrix
from numpy import linalg

#Written by N. Stein 10/12/18
#Builds off the PLA from HW1 to solve a d = 2 problem using linear regression

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
        x = data[i][:]
        y = data[i][3]
        if evaluate_sign(w, x) != y:
            # then the point is misclassified
            misclassified_points.append(x)
    return misclassified_points

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
    n_iterations = 100 # number of iterations of PLA
    n_points_Eout = 1000
    data, f = generate_data(n_points)

    # fill the x and y matrices
    X = np.empty((n_points, 3), dtype=float)
    Y = np.empty((n_points, 1), dtype=float)

    data_arr = np.array(data)
    X = data_arr[:, 0:3]
    Y = data_arr[:, 3]

    XTX = np.matmul(X.T, X)
    invXTX = np.linalg.inv(XTX)
    invXTXT = np.matmul(invXTX, X.T)
    w = np.matmul(invXTXT, Y)
    w = w.tolist()
    w.append(0)

    # check number of miscalssified points (for in sample error)
    misclassified_points = check_classification(data, w)
    n_misclassified = len(misclassified_points)

    # find out of sample error
    # generate n_points_Eout data points and check how well they are classified
    data_Eout = []
    for j in range(n_points_Eout):
        pt = [1,random.uniform(-1, 1), random.uniform(-1, 1), 1]  # randomly generated x1 and x2 and choose y = 1
        pt[3] = evaluate_sign_initial(f, pt)  # evaluate point to select the correct y value for the target function f
        data_Eout.append(pt)
    misclassified_out = check_classification(data_Eout, w)
    n_misclassified_out = len(misclassified_out)

    # after finding the weights using linear regression, we use them
    # as a vector of initial weights for PLA
    w_initial = w
    w_final = w_initial
    for i in range(n_iterations):
        if i == 0:
            misclassified_points = check_classification(data, w_initial)

        # choose a random point out of the set of misclassified points and use it to update w
        if(misclassified_points):  # only do this if there are misclassified points
            random_wrong_point = random.choice(misclassified_points)
            w_final[0] = w_final[0] + random_wrong_point[0]*random_wrong_point[3]
            w_final[1] = w_final[1] + random_wrong_point[1]*random_wrong_point[3]
            w_final[2] = w_final[2] + random_wrong_point[2]*random_wrong_point[3]
            misclassified_points = check_classification(data, w_final)

        # find the number of iterations it takes until all points are properly classified
        n_iterations_to_converge = 0
        if len(misclassified_points) == 0:
            n_iterations_to_converge = i

    return f, w, n_misclassified, n_misclassified_out, n_points, n_points_Eout, n_iterations_to_converge, n_iterations

def n_misclassified(f, w_final, n_points_to_check, number_different):
    for i in range(n_points_to_check):
        pt = [1,random.uniform(-1, 1), random.uniform(-1, 1), 1]  # randomly generated x1 and x2 and choose y = 1
        f_eval = evaluate_sign_initial(f, pt)  #evaluate points on f
        w_eval = evaluate_sign(w_final, pt)    #evaluate points on w_final
        if f_eval != w_eval: number_different[q] += 1
    return number_different


# average results over many runs
index = []
n_runs = 5000
n_points_to_check = 1000
sum_misclassified = 0
sum_misclassified_out = 0
sum_n_iterations_to_converge = 0
for q in range(n_runs):
    f, w, n_misclassified, n_misclassified_out, n_points, n_points_Eout, n_iterations_to_converge, n_iterations = main()
    print(q)
    sum_misclassified += n_misclassified
    sum_misclassified_out += n_misclassified_out
    sum_n_iterations_to_converge += n_iterations_to_converge
    # Now check to see how well w classifies the points

print(sum_misclassified/n_runs/n_points) # answer to question 5
print(sum_misclassified_out/n_runs/n_points_Eout) # answer to question 6
print(sum_n_iterations_to_converge/n_runs/n_iterations)
