import numpy as np

#HW 7 problem 7
error_constant = 0
error_linear = 0
for q in range(3):
    #p = np.sqrt(np.sqrt(3)+4)
    #p = np.sqrt(np.sqrt(3)-1)
    #p = np.sqrt(9 + 4*np.sqrt(6))
    p = np.sqrt(9 - np.sqrt(6))

    din = [[-1,0],[p,1],[1,0]]
    dout = [din[q]]
    del din[q]

    # constant function case
    b = 1/2*(din[0][1] + din[1][1])
    error_constant += (abs(dout[0][1] - b))**2

    #linear function case
    a = (din[1][1] - din[0][1])/(din[1][0] - din[0][0])
    b = din[0][1] - a*din[0][0]
    error_linear += (abs(dout[0][1] - a*dout[0][0] - b))**2

print('Constant error for rho = ', p, ' is ', error_constant/3)
print('Linear error for rho = ', p, ' is ', error_linear/3)