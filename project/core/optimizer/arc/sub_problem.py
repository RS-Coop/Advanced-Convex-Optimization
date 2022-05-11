'''
ARC sub-problem solvers.

Here we implement methods for solving the regularized cubic minimization
problem for ARC. A number of methods are implemented to be chosen at the users
discretion.
'''

import numpy as np
import numpy.linalg as la
from numpy.random import default_rng
from scipy.optimize import minimize

'''
Sub-problem objective function
'''
def cubic(s, g, H, sigma):
    return g.T@s + 0.5*s.T@H@s + (sigma/3)*(la.norm(s)**3)

def cubicGrad(s, g, H, sigma):
    return g + H@s + sigma*la.norm(s)*s

'''
Dispatcher for solution methods
'''
def arcSub(method, args, kw={}):
    solver_map = {
        'lanczos' : lanczos
    }

    return solver_map[method](*args, **kw)

'''
Solvers: All have the following structure.
Input:
    g -> Gradient
    H -> Hessian
    sigma -> Regularization parameter
    maxitr -> Maximum iterations (optional)
    tol -> Tolerance (optional)
'''

'''
Generalized Lanczos method as described in the Non-Convex Newton and Inexact
Newton papers.

Run Lanczos iterations and then solve low-dim cubic problem with tri-diagonal
matrix.
'''
def lanczos(g, H, sigma, maxitr=500, tol=1e-6):
    d = g.shape[0]
    K = min(d, maxitr)
    Q = np.zeros((d, K))

    q = g
    q = q/la.norm(q)

    T = np.zeros((K+1, K+1))

    tol = min(tol, tol*la.norm(g))

    for i in range(K):
        Q[:,i] = q
        v = H@q
        T[i,i] = q.T@v

        #Orthogonalize
        r = v - Q[:,:i]@(Q[:,:i].T@v)

        b = la.norm(r)
        T[i,i+1] = b
        T[i+1,i] = b

        if b < tol:
            q=0
            break

        q = r/b

    #Compute last diagonal element
    T[i, i] = q.T@H@q

    T = T[:i, :i]
    Q = Q[:, :i]

    if la.norm(T) < tol and la.norm(g) < np.spacing(1):
        return zeros(d), 0

    gt = Q.T@g

    #Optimization inception
    z0 = np.zeros((i,1))
    out = minimize(cubic, z0, jac=cubicGrad,
                    args=(gt, T, sigma), method='BFGS', tol=tol)
    z = out['x']
    m = out['fun']

    return Q@z, m
