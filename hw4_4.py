import random
import numpy as np
import matplotlib.pyplot as plt

#generate the target function
def target_function(x):
    f = np.sin(np.pi * x)
    return f

#randomly choose two points from the target function
def choose_points(x, f):
    px = random.sample(list(enumerate(x)), 2)
    px1 = px[0][1]
    px2 = px[1][1]
    pf1 = f[px[0][0]]
    pf2 = f[px[1][0]]
    return px1, px2, pf1, pf2

# calculate the bias using the average g that was found previously for the case h(x) = b
def find_bias_b(f, b_average):
    bias = []
    for i in range(len(f)):
        bias.append((f[i] - b_average) ** 2)
    return bias

# find the variance of each hypothesis relative to the average g that was found previously for the case h(x) = ax
def find_variance_b(b_new, b_average):
    variance_single_point = []
    variance_single_point.append((b_new - b_average) ** 2)
    return variance_single_point

# calculate the bias using the average g that was found previously for the case h(x) = ax
def find_bias_ax(x, f, a_average):
    bias = []
    for i in range(len(f)):
        bias.append((f[i] - a_average * x[i]) ** 2)
    return bias

# find the variance of each hypothesis relative to the average g that was found previously for the case h(x) = ax
def find_variance_ax(x, a_new, a_average):
    variance_single_point = []
    for i in range(len(x)):
        variance_single_point.append((a_new * x[i] - a_average * x[i]) ** 2)
    return variance_single_point

# calculate the bias using the average g that was found previously for the case h(x) = ax + b
def find_bias_ax_plus_b(x, f, a_average_caseC, b_average_caseC):
    bias = []
    for i in range(len(f)):
        bias.append((f[i] - a_average_caseC * x[i] - b_average_caseC) ** 2)
    return bias

# find the variance of each hypothesis relative to the average g that was found previously for the case h(x) = ax
def find_variance_ax_plus_b(x, a_ax_plus_b_new, b_ax_plus_b_new, a_average_caseC, b_average_caseC):
    variance_single_point = []
    for i in range(len(x)):
        variance_single_point.append((a_ax_plus_b_new * x[i] + b_ax_plus_b_new - a_average_caseC * x[i] - b_average_caseC) ** 2)
    return variance_single_point

# calculate the bias using the average g that was found previously for the case h(x) = ax^2
def find_bias_ax_squared(x, f, a_average_ax_squared):
    bias = []
    for i in range(len(f)):
        bias.append((f[i] - a_average_ax_squared ** x[i]) ** 2)
    return bias

# find the variance of each hypothesis relative to the average g that was found previously for the case h(x) = ax^2
def find_variance_ax_squared(x, a_ax_squared_new, a_ax_squared_average):
    variance_single_point = []
    for i in range(len(x)):
        variance_single_point.append((a_ax_squared_new * x[i] ** 2 - a_ax_squared_average * x[i] ** 2) **  2)
    return variance_single_point

# find the bias of each hypothesis relative to the average g that was found previously for the case h(x) = ax^2 + b
def find_bias_ax_squared_plus_b(x, f, a_average_ax_squared_plus_b, b_average_ax_squared_plus_b):
    bias = []
    for i in range(len(f)):
        bias.append((f[i] - a_average_ax_squared_plus_b * x[i] ** 2 - b_average_ax_squared_plus_b) ** 2)
    return bias

# find the variance of each hypothesis relative to the average g that was found previously for the case h(x) = ax^2 + b
def find_variance_ax_squared_plus_b(x, a_new, b_new, a_average, b_average):
    variance_single_point = []
    for i in range(len(x)):
        variance_single_point.append((a_new * x[i] ** 2 + b_new - a_average * x[i] ** 2 - b_average) ** 2)
    return variance_single_point

def main():
    n_runs = 10000
    x = np.arange(-1, 1, .001)
    f = target_function(x)
    a = []
    b = []
    a_caseC = []
    b_caseC = []
    a_ax_squared = []
    a_ax_squared_plus_b = []
    b_ax_squared_plus_b = []
    a_average = 1.43
    b_average = 0
    a_average_caseC = 0.76
    b_average_caseC = 0
    a_average_ax_squared = 0.16
    a_average_ax_squared_plus_b = 1.86
    b_average_ax_squared_plus_b = 18.82
    variance_ax = 0
    variance_b = 0
    variance_ax_plus_b = 0
    variance_ax_squared = 0
    variance_caseE = 0
    for i in range(n_runs):
        px1, px2, pf1, pf2 = choose_points(x, f)

        # For case h(x) = b
        b_new = (pf1 + pf2)/2
        b.append([b_new])
        bias_b = find_bias_b(f, b_average)
        variance_single_b = find_variance_b(b_new, b_average)
        variance_b += np.average(variance_single_b)

        # For case h(x) = ax
        a_ax_new = (px1 * pf1 + px2 * pf2) / (px1 ** 2 + px2 ** 2)
        a.append([a_ax_new])
        bias_ax = find_bias_ax(x, f, a_average)
        variance_single_ax = find_variance_ax(x, a_ax_new, a_average)
        variance_ax += np.average(variance_single_ax)

        # For case h(x) = ax + b
        a_ax_plus_b_new = (pf2 - pf1)/(px2 - px1)
        a_caseC.append([a_ax_plus_b_new])
        b_ax_plus_b_new = pf1 - a_ax_plus_b_new*px1
        b_caseC.append([b_ax_plus_b_new])
        bias_caseC = find_bias_ax_plus_b(x, f, a_average_caseC, b_average_caseC)
        variance_single_caseC = find_variance_ax_plus_b(x, a_ax_plus_b_new, b_ax_plus_b_new, a_average_caseC, b_average_caseC)
        variance_ax_plus_b += np.average(variance_single_caseC)

        # For case h(x) = ax^2
        a_ax_squared_new = ((px1 ** 2)*pf1 + (px2 ** 2)*pf2)/(px1 ** 4 + px2 ** 4)
        a_ax_squared.append([a_ax_squared_new])
        bias_ax_squared = find_bias_ax_squared(x, f, a_average_ax_squared)
        variance_single_ax_squared = find_variance_ax_squared(x, a_ax_squared_new, a_average_ax_squared)
        variance_ax_squared += np.average(variance_single_ax_squared)

        # For case h(x) = ax^2 + b
        a_ax_squared_plus_b_new = (2*((px1 ** 2)*pf1 + (px2 ** 2)*pf2) - (px1 ** 2 - px2 ** 2)*(pf1 + pf2))/(2*(px1 ** 4 + px2 ** 4) - (px1 ** 2 + px2 ** 2) ** 2)
        a_ax_squared_plus_b.append([a_ax_squared_plus_b_new])
        b_ax_squared_plus_b_new = (pf1 + pf2 - a_ax_squared_plus_b_new*(px1 ** 2 + px2 ** 2))/2
        b_ax_squared_plus_b.append([b_ax_squared_plus_b_new])
        bias_caseE = find_bias_ax_squared_plus_b(x, f, a_average_ax_squared_plus_b, b_average_ax_squared_plus_b)
        variance_single_caseE = find_variance_ax_squared_plus_b(x, a_ax_squared_plus_b_new, b_ax_squared_plus_b_new, a_average_ax_squared_plus_b, b_average_ax_squared_plus_b)
        variance_caseE += np.average(variance_single_caseE)

    return x, f, a, b, a_caseC, b_caseC, bias_ax, variance_ax, bias_b, variance_b, bias_caseC, variance_ax_plus_b, a_ax_squared, bias_ax_squared, variance_ax_squared, a_ax_squared_plus_b, b_ax_squared_plus_b, bias_caseE, variance_caseE, n_runs

x, f, a, b, a_average_caseC, b_average_caseC, bias_ax, variance_ax, bias_b, variance_b, bias_caseC, variance_ax_plus_b, a_ax_squared, bias_ax_squared, variance_ax_squared, a_ax_squared_plus_b, b_ax_squared_plus_b, bias_caseE, variance_caseE, n_runs = main()
print("average value of a in h(x) = ax:", np.average(a)) # average value of a
print("average value of b in h(x) = b", np.average(b)) # average value of b
print("average value of a in h(x) = ax + b", np.average(a_average_caseC)) # average value of a in h(x) = ax + b
print("average value of b in h(x) = ax + b", np.average(b_average_caseC)) # average value of b in h(x) = ax + b
print("average value of a in h(x) = ax^2", np.average(a_ax_squared)) # average value of a in h(x) = ax^2
print("average value of a in h(x) = ax^2 + b", 1.86) # average value of a in h(x) = ax^2 + b
print("average value of b in h(x) = ax^2 + b", 18.82) # average value of b in h(x) = ax^2 + b
print("average bias of h(x) = ax", np.average(bias_ax)) # average bias of h(x) = ax
print("average bias of h(x) = ax", variance_ax/n_runs)
print("expected value of out-of-sample error for h(x) = ax", variance_ax/n_runs + np.average(bias_ax)) # expected value of out-of-sample error for h(x) = ax
print("expected value of out-of-sample error for h(x) = b", variance_b/n_runs + np.average(bias_b)) # expected value of out-of-sample error for h(x) = b
print("expected value of out-of-sample error for h(x) = ax + b", variance_ax_plus_b/n_runs + np.average(bias_caseC)) # expected value of out-of-sample error for h(x) = ax + b
print("expected value of out-of-sample error for h(x) = ax^2", variance_ax_squared/n_runs + np.average(bias_ax_squared)) # expected value of out-of-sample error for h(x) = ax^2
print("expected value of out-of-sample error for h(x) = ax^2 + b", 2671184) # expected value of out-of-sample error for h(x) = ax^2 + b

#plot things up
#plt.plot(x,f)
#plt.show()