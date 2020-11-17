from scipy.sparse.linalg import *
import scipy as sp
import numpy as np
import pyamg                   
 
A = sp.sparse.eye(10, format='csr')
b = np.ones(A.shape[0])

ml = pyamg.ruge_stuben_solver(A)            
M = ml.aspreconditioner()

x = cg(A, b, M=M)