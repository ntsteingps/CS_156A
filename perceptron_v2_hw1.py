import random
import numpy as np
import matplotlib.pyplot as plt

#Written by N. Stein 10/7/18

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
    w_initial = [0, 0, 0, 0]
    w_final   = w_initial
    n_points = 10  # number of training points
    n_iterations = 5000  # number of iterations
    data, f = generate_data(n_points)
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

        # print(w_final, len(misclassified_points),i)
        if len(misclassified_points) == 0:
            # linarray = np.linspace(-1,1,100)
            # plt.plot(linarray,w_final[0]*linarray + w_final[1],'r--')
            # plt.xlim(-1,1)
            # plt.ylim(-1,1)
            # print(i, f, w_final)
            return i, f, w_final, data#, data, f # set output to i to calculate the average number of iterations for convergence

    #print(w_final)
    #return f, w_final

def n_misclassified(f, w_final, n_points_to_check, number_different):
    for i in range(n_points_to_check):
        pt = [1,random.uniform(-1, 1), random.uniform(-1, 1), 1]  # randomly generated x1 and x2 and choose y = 1
        f_eval = evaluate_sign_initial(f, pt)  #evaluate points on f
        w_eval = evaluate_sign(w_final, pt)    #evaluate points on w_final
        if f_eval != w_eval: number_different[q] += 1
    return number_different


# average results over many runs
index = []
n_runs = 1000
n_points_to_check = 1000 # points to check for question 8
for q in range(n_runs):
    print(q)
    index_val, f, w_final, data = main()
    #print(w_final)
    index.append(index_val)
    for i in range(len(data)):
        if data[i][3] == -1:
            plt.plot(data[i][1], data[i][2], 'go')
        else:
            plt.plot(data[i][1], data[i][2], 'ro')
    #print(index_val)

    # now generate many points and classify them according to f and w to see how well w performs
    number_different = np.empty((n_points_to_check))
    number_different = n_misclassified(f, w_final, n_points_to_check, number_different)

# print the number of iterations it took until there were no misclassified points (average_index) and the average
# of that number over many runs
average_index = sum(index) / float(len(index))
print(average_index)
print(sum(number_different)/n_runs/n_points_to_check)

#plot the target function with the points
# linarray = np.linspace(-1,1,100)
# plt.plot(linarray,f[1]*linarray + f[2],'b--')
# plt.plot(linarray,-(w_final[0]/w_final[2])*linarray/(w_final[0]/w_final[1]) - w_final[0]/w_final[2], 'r--')
# plt.xlim(-1,1)
# plt.ylim(-1,1)
# plt.show()

