import numpy as np
from svmutil import *
from sklearn.model_selection import train_test_split

#Import data from txt files
data_train = np.genfromtxt('in_hw8.txt')
data_test  = np.genfromtxt('out_hw8.txt')

#separate into two lists where one contains the binary classification
def populate_lists(data, index):
    x = data[:, 1:]
    y = data[:,0]
    new_y = []
    for i in range(len(data)):
        if index == y[i]:
            new_y.append(1)
        else:
            new_y.append(-1)
    return x.tolist(), new_y

n = 10 # number of possible values to check
# for i in range(n):
#     x, y = populate_lists(data_train, i)
#     problem = svm_problem(y, x)
#     parameters = svm_parameter('-h 0 -t 1 -c .01 -d 2 -g 1 -r 1')
#     model = svm_train(problem, parameters)
#     p_labels, p_accs, p_vals = svm_predict(y, x, model)
#
#     #Compute E_in using the accuracy (first index in the output tuple p_accs
#     E_in = 100 - p_accs[0]
#     print('E_in of ', i,' is: ', E_in)

# Test with classifiers of only 1 or 5 for questions 5 and 6

i = 1
qval = 5
idx_1 = np.argwhere(data_train[:,0] == 1)
idx_5 = np.argwhere(data_train[:,0] == 5)
idx_1_test = np.argwhere(data_test[:,0] == 1)
idx_5_test = np.argwhere(data_test[:,0] == 5)

data_1_5 = data_train[np.append(idx_1, idx_5)]
y_1_5 = data_1_5[:, 0]
new_y_1_5 = []
for q in range(len(y_1_5)):
    if 1 == y_1_5[q]:
        new_y_1_5.append(1)
    else:
        new_y_1_5.append(-1)
x_1_5 = data_1_5[:, 1:]
x_1_5 = x_1_5.tolist()

test_data_1_5 = data_test[np.append(idx_1_test, idx_5_test)]
test_y_1_5 = test_data_1_5[:, 0]
new_test_y_1_5 = []
for q in range(len(test_y_1_5)):
    if 1 == test_y_1_5[q]:
        new_test_y_1_5.append(1)
    else:
        new_test_y_1_5.append(-1)
test_x_1_5 = test_data_1_5[:, 1:]
test_x_1_5 = test_x_1_5.tolist()

problem = svm_problem(new_y_1_5, x_1_5)
parameters = svm_parameter('-h 0 -t 1 -c .01 -d 5 -g 1 -r 1')
model = svm_train(problem, parameters)
p_labels_train, p_accs_train, p_vals_train = svm_predict(new_y_1_5, x_1_5, model)
p_labels_test , p_accs_test , p_vals_test  = svm_predict(new_test_y_1_5, test_x_1_5, model)

#Compute E_in using the accuracy (first index in the output tuple p_accs
E_in = 100 - p_accs_train[0]
E_out = 100 - p_accs_test[0]

#Print output
print('E_in of ', i,' is: ', E_in)
print('E_out of ', i,' is: ', E_out)
print('C = ', .01)
print('Q = ', qval)

# Now we will implement 10-fold cross validation for the polynomial kernel
n_runs = 100
C = [1e-4, 1e-3, 1e-2, 1e-1, 1]
selected_C = []
selected_E = []

for i in range(n_runs):
    E_val_tracker = []
    for k, Cval in enumerate(C):
        xtrain, xtest, ytrain, ytest = train_test_split(x_1_5, new_y_1_5, test_size = 0.1)
        problem = svm_problem(ytrain, xtrain)
        parameters = svm_parameter('-h 0 -t 1 -c {} -d 2 -g 1 -r 1'.format(Cval))
        model = svm_train(problem, parameters)

        p_labels, p_accs, pvals = svm_predict(ytest, xtest, model)
        E_val_tracker.append(100-p_accs[0])
    selected_C.append(C[E_val_tracker.index(min(E_val_tracker))])
    selected_E.append(min(E_val_tracker))

print(selected_C)
print('The average value of E_cv is', sum(selected_E)/n_runs)
print('C = 0.0001 is selected ', 100*sum(i == 1e-4 for i in selected_C)/n_runs, '% of the time')
print('C = 0.001 is selected ', 100*sum(i == 1e-3 for i in selected_C)/n_runs, '% of the time')
print('C = 0.01 is selected ', 100*sum(i == 1e-2 for i in selected_C)/n_runs, '% of the time')
print('C = 0.1 is selected ', 100*sum(i == 1e-1 for i in selected_C)/n_runs, '% of the time')
print('C = 1 is selected ', 100*sum(i == 1e-0 for i in selected_C)/n_runs, '% of the time')

#Problem 8
E_val_tracker = []
for i in range(n_runs):
    xtrain, xtest, ytrain, ytest = train_test_split(x_1_5, new_y_1_5, test_size=0.1)
    problem = svm_problem(ytrain, xtrain)
    parameters = svm_parameter('-h 0 -t 1 -c .001 -d 2 -g 1 -r 1')
    model = svm_train(problem, parameters)

    p_labels, p_accs, pvals = svm_predict(ytest, xtest, model)
    E_val_tracker.append(100 - p_accs[2])
print(E_val_tracker)
print('The average value of E_cv is', np.mean(E_val_tracker))

#Problems 9 and 10
#Now we consider the radial basis function kernel K(x_n, x_m) = exp(-||x_n - x_m||^2) in the soft-margin SVM approach
C = [1e-2, 1, 1e2, 1e4, 1e6]
Ein = []
Eout = []

for k, Cval in enumerate(C):
    problem = svm_problem(ytrain, xtrain)
    parameters = svm_parameter('-g 1 -c {}'.format(Cval))
    model = svm_train(problem, parameters)
    p_labels_train, p_accs_train, pvals_train = svm_predict(ytrain, xtrain, model)
    p_labels_test, p_accs_test, pvals_test = svm_predict(ytest, xtest, model)
    print('E_in for C = {} is '.format(Cval), 100 - p_accs_train[0])
    print('E_out for C = {} is '.format(Cval), 100 - p_accs_test[0])
