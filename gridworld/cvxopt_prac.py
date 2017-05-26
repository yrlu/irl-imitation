from cvxopt import matrix, solvers
import numpy as np


# Linear program example:
# min     2x_1 + x_2 
# s.t.    -x_1 + x_2 <= 1
#         -x_1 - x_2 <= 2
#              - x_2 <= 0
#         x_1 - 2x_2 <= 4
# 
# min     c^Tx 
# s.t.    Ax <= b

A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])
b = matrix([ 1.0, -2.0, 0.0, 4.0 ])
c = matrix([ 2.0, 1.0 ])
sol=solvers.lp(c,A,b)

print sol['x']


# Quadratic program example
# min    2x_1^2 + x_2^2 + x_1x_2 + x_1 + x_2 
# s.t.   -x_1       <= 0
#             -x_2  <= 0
#        x_1 + x_2  = 1

# min    1/2*x^TQx + p^Tx
# s.t.   Gx <= h
#        Ax  = b
Q = 2*matrix([ [2, .5], [.5, 1] ])
p = matrix([1.0, 1.0])
G = matrix([[-1.0,0.0],[0.0,-1.0]])
h = matrix([0.0,0.0])
A = matrix([1.0, 1.0], (1,2))
b = matrix(1.0)
sol=solvers.qp(Q, p, G, h, A, b)
print sol['x']


# Use matrices
A = matrix([1.0, 1.0], (1,2))
print A

A = [[1.0, 1.0]]
A = np.array(A)
print A
A = matrix(A)
print A


