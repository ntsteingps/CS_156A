import random
import numpy as np
import matplotlib.pyplot as plt

#Written by N. Stein 11/12/18

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

def generate_data(num_points, din):
    f = [0, 0, 0, 0, 0, 0, 0, 0]
    data_points = []
    # Read in data
    for j in range(num_points):
        # read in data for in sample points
        pt = [1, din[j][0], din[j][1], 0, 0, 0, 0, 0, 0]  # randomly generated x1 and x2 and choose y = 1
        pt[3] = pt[1]**2
        pt[4] = pt[2]**2
        pt[5] = pt[1]*pt[2]
        pt[6] = abs(pt[1] - pt[2])
        pt[7] = abs(pt[1] + pt[2])
        pt[8] = din[j][2] # read in y-val
        data_points.append(pt)
    return data_points, f

# takes in vector with training data and adds a noise component (flips sign) of specified fraction of points
def add_noise(data, fraction, n_points):
    noisy_data = data
    for i in range(round(fraction*n_points)):
        noisy_data[i][len(data[0])-1] = -1*data[i][len(data[0])-1]
    return noisy_data

# Solve the linear regression problem with regularization
def main():
    noise_fraction = 0.1 # add noise to this proportion of data
    reg_switch = 1 # denotes whether we are performing regularization
    k = -2
    lam = 10**k
    din = np.genfromtxt('in_hw6.txt')
    dout = np.genfromtxt('out_hw6.txt')
    n_points = len(din)
    data, f = generate_data(n_points, din)
    #data = add_noise(data, noise_fraction, n_points)

    # fill the x and y matrices
    X = np.empty((n_points, 3), dtype=float)
    Y = np.empty((n_points, 1), dtype=float)

    data_arr = np.array(data)
    X = data_arr[:, 0:len(data[0])-1]
    Y = data_arr[:, len(data[0])-1]

    reg_term = lam*np.identity(len(X[0,:])) # regularization term
    XTX = np.matmul(X.T, X) + reg_term*reg_switch
    invXTX = np.linalg.inv(XTX)
    invXTXT = np.matmul(invXTX, X.T)
    w = np.matmul(invXTXT, Y)
    w = w.tolist()
    w.append(0)

    # check number of misclassified points (for in sample error)
    misclassified_points = check_classification(data, w)
    num_misclassified = len(misclassified_points)
    E_in = num_misclassified/len(din) # In sample error
    print(E_in)

    #Calculate out of sample error with given points
    data_out, f = generate_data(len(dout), dout)
    misclassified_points = check_classification(data_out, w)
    num_misclassified = len(misclassified_points)
    E_out = num_misclassified/len(dout)
    print(E_out)

    return f, w, data, num_misclassified, n_points

def n_misclassified(w_final, n_points_to_check, noise_fraction, din):
    n_different = 0
    pt, f = generate_data(n_points_to_check, din)
    pt = add_noise(pt, noise_fraction, n_points_to_check)
    for i in range(n_points_to_check):
        f_eval = pt[i][len(pt[0])-1] # evaluate points on f for f(x1,x2) = sign(x1^2 + x2^2 - 0.6)
        w_eval = evaluate_sign(w_final, pt[i])    #evaluate points on w_final
        if f_eval != w_eval: n_different += 1
    return n_different

main()



