import random
import numpy as np
import matplotlib.pyplot as plt

#Written by N. Stein 10/13/18

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
        y = data[i][len(data[0])-1]
        if evaluate_sign(w, x) != y:
            # then the point is misclassified
            misclassified_points.append(x)
    return misclassified_points

def generate_data(num_points):
    # generate target function f (w) by choosing a random line in the target function
    # we will take two random, uniformly distributed points in [-1,1]x[-1,1] and take the line passing through them
    #point1 = [random.uniform(-1, 1), random.uniform(-1, 1)]
    #point2 = [random.uniform(-1, 1), random.uniform(-1, 1)]
    #f = [0,0,0]
    #f[1] = (point2[1]-point1[1])/(point2[0]-point1[0])  # slope
    #f[2] = point1[1] - f[1]*point1[0]  # y int
    f = [0, 0, 0, 0, 0, 0]

    # choose the inputs x_n of the data set as random points uniformly in x
    data_points = []
    for j in range(num_points):
        pt = [1, random.uniform(-1, 1), random.uniform(-1, 1), 0, 0, 0, 0]  # randomly generated x1 and x2 and choose y = 1
        pt[3] = pt[1]*pt[2]
        pt[4] = pt[1]**2
        pt[5] = pt[2]**2
        pt[6] = np.sign(pt[1]**2 + pt[2]**2 - 0.6)  # evaluate point on the function f(x1,x2) = sign(x1^2 + x2^2 - 0.6)
        data_points.append(pt)
    return data_points, f

# takes in vector with training data and adds a noise component (flips sign) of specified fraction of points
def add_noise(data, fraction, n_points):
    noisy_data = data
    for i in range(round(fraction*n_points)):
        noisy_data[i][len(data[0])-1] = -1*data[i][len(data[0])-1]
    return noisy_data

# Start the PLA with the weight vector w being all zeros

def main():
    n_points = 1000  # number of training points
    noise_fraction = 0.1 # add noise to this proportion of data
    data, f = generate_data(n_points)
    data = add_noise(data, noise_fraction, n_points)

    # fill the x and y matrices
    X = np.empty((n_points, 3), dtype=float)
    Y = np.empty((n_points, 1), dtype=float)

    data_arr = np.array(data)
    X = data_arr[:, 0:len(data[0])-1]
    Y = data_arr[:, len(data[0])-1]

    XTX = np.matmul(X.T, X)
    invXTX = np.linalg.inv(XTX)
    invXTXT = np.matmul(invXTX, X.T)
    w = np.matmul(invXTXT, Y)
    w = w.tolist()
    w.append(0)

    # check number of misclassified points (for in sample error)
    misclassified_points = check_classification(data, w)
    num_misclassified = len(misclassified_points)

    return f, w, data, num_misclassified, n_points

def n_misclassified(w_final, n_points_to_check, noise_fraction):
    n_different = 0
    pt, f = generate_data(n_points_to_check)
    pt = add_noise(pt, noise_fraction, n_points_to_check)
    for i in range(n_points_to_check):
        f_eval = pt[i][len(pt[0])-1] # evaluate points on f for f(x1,x2) = sign(x1^2 + x2^2 - 0.6)
        w_eval = evaluate_sign(w_final, pt[i])    #evaluate points on w_final
        if f_eval != w_eval: n_different += 1
    return n_different

# average results over many runs
n_runs = 1000
n_points_to_check = 1000 # points to check for question 8
noise_fraction = 0.1
number_different = np.empty((n_runs))
w_final_sum = [0, 0, 0, 0, 0, 0, 0]
w_final_super_list = [] # list of all w_finals
sum_misclassified = 0
n_amatch = 0
n_bmatch = 0
n_cmatch = 0
n_dmatch = 0
n_ematch = 0
for q in range(n_runs):
    print(q)
    f, w_final, data, num_misclassified, n_points = main()
    w_final_sum = [x + y for x, y in zip(w_final_sum, w_final)] #add w_final in each iteration
    w_final_super_list.append(w_final)

    #classify point according to w and check how well it agrees with the other hypotheses
    pt = [random.uniform(-1, 1), random.uniform(-1, 1)]
    w_val = np.sign(w_final[0] + w_final[1]*pt[0] + w_final[2]*pt[1] + w_final[3]*pt[0]*pt[1] + w_final[4]*pt[0]**2 + w_final[5]*pt[1]**2)
    a_val = np.sign(-1 + -.05*pt[0] + .08*pt[1] + .13*pt[0]*pt[1] + 1.5*pt[0]**2 + 1.5*pt[1]**2)
    b_val = np.sign(-1 + -.05*pt[0] + .08*pt[1] + .13*pt[0]*pt[1] + 1.5*pt[0]**2 + 15*pt[1]**2)
    c_val = np.sign(-1 + -.05*pt[0] + .08*pt[1] + .13*pt[0]*pt[1] + 15*pt[0]**2 + 1.5*pt[1]**2)
    d_val = np.sign(-1 + -1.5*pt[0] + .08*pt[1] + .13*pt[0]*pt[1] + .05*pt[0]**2 + .05*pt[1]**2)
    e_val = np.sign(-1 + -.05*pt[0] + .08*pt[1] + 1.5*pt[0]*pt[1] + .15*pt[0]**2 + .15*pt[1]**2)
    n_amatch += (a_val == w_val)
    n_bmatch += (b_val == w_val)
    n_cmatch += (c_val == w_val)
    n_dmatch += (d_val == w_val)
    n_ematch += (e_val == w_val)

    # now generate many points and classify them according to f and w to see how well w performs
    number_different[q] = n_misclassified(w_final, n_points_to_check, noise_fraction)

    # check in sample error
    sum_misclassified += num_misclassified

# print the number of iterations it took until there were no misclassified points (average_index) and the average
# of that number over many runs
print(sum_misclassified/n_runs/n_points) # in sample rror
print(sum(number_different)/n_runs/n_points_to_check) # out of sample error
print(n_amatch/n_runs)
print(n_bmatch/n_runs)
print(n_cmatch/n_runs)
print(n_dmatch/n_runs)
print(n_ematch/n_runs)

