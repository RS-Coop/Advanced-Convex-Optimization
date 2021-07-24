'''
Utility functions including data loading, loss functions,
gradient validation, etc.
'''
import warnings

import numpy as np
import numpy.linalg as lg
import pandas as pd
import scipy.special as sps
from sklearn.model_selection import train_test_split

'''DATA LOADING'''

'''
Implements functionality to load a basic dataset using pandas.

NOTE: At the moment this is pretty specific to the spambase dataset. Not
sure of the behaviour on other inputs.

Input:


Output:
    X, y
'''
def loadDataset(file_path, data_map, label_map, pd_kw={}, tts_kw={}):
    data = pd.read_csv(file_path, **pd_kw)

    if data.isnull().sum().sum() != 0:
        warnings.warn('Data has missing values!')

    X = data.iloc[:,0:-2]
    y = data.iloc[:,-1]

    X = X.apply(data_map)
    y = y.apply(label_map)
    
    return train_test_split(X, y, **tts_kw)

'''Loss Functions'''

'''
Negative Log-Likelihood function, gradient, and hessian.

w -> px1 column vector
X -> nxp matrix
y -> nx1 column vector
'''

#returns scalar
def nLogLike(w, X, y):
    l = np.logaddexp(np.zeros(y.shape), -y*X@w)
    return np.sum(l)

#returns vector
def nLogLike_grad(w, X, y):
    u = sps.expit(-y*(X@w))
    
    return -X.T@(y*u)

#returns matrix
def nLogLike_hess(w, X, y):
    u = sps.expit(-y*(X@w))
    S = np.diag(u*(1-u))
    
    return X.T@S@X


'''Gradient Checking'''

'''
Implements functionality to validate that the gradient of a given function
is indeed correct.

Initialization:
    f -> function
    grad -> function gradient
'''
class GradChecker():
    def __init__(self, f, grad):
        self.f = f
        self.grad = grad

        self.dispatcher = {
            'dolfin': self.dolfinAdj,
            'complex': self.complexStep,
        }

    def reset(self, f, grad):
        self.f = f
        self.grad = grad

    def check(self, method='dolfin', x0=None, dim=None, kwargs={}):
        checker = self.dispatcher[method]
        self.kw = kwargs

        rng = np.random.default_rng()

        if dim is not None:
            x0 = rng.integers(0,100,(dim,1)) #Random vector

        checker(x0)

    '''
    Dolfin Adjoint: Assumes hessian is smooth
    '''
    def dolfinAdj(self, x0):
        valid = True

        rng = np.random.default_rng()

        x1 = x0*rng.random(x0.shape) #random vector
        f = self.f(x0, **self.kw)

        h = np.geomspace(1e-7, 1.0, 8)

        for i in range(7,0,-1):
            f0 = self.f(x0 + h[i]*x1, **self.kw)
            e0 = np.abs(f + np.vdot(self.grad(x0, **self.kw), h[i]*x1) - f0)

            f1 = self.f(x0 + h[i-1]*x1, **self.kw)
            e1 = np.abs(f + np.vdot(self.grad(x0, **self.kw), h[i-1]*x1) - f1)

            if e0!=0 and e1!=0:
                order = np.rint(np.log(e1/e0) / np.log(h[i-1]/h[i]))
                if order != 2:
                    print('Order violated: {}'.format(order))
                    print('Error k,k+1: {},{} \n'.format(e0, e1))

                    valid = False

        print('Gradient is valid: {} \n'.format(valid))

    '''
    Complex-Step Derivative: Assuming a few things related to complex numbers
    e.g. analyticity of f.
    '''
    def complexStep(self, x0):
        valid = True

        g0 = self.grad(x0, **self.kw)

        h = np.geomspace(1e-7, 1.0, 8)

        for i in range(7,0,-1):
            e0 = np.abs(np.sum(g0) - np.imag(self.f(x0 + h[i]*1j, **self.kw) / h[i]))

            e1 = np.abs(np.sum(g0) - np.imag(self.f(x0 + h[i-1]*1j, **self.kw) / h[i-1]))


            if e0!=0 and e1!=0:
                order = np.rint(np.log(e1/e0) / np.log(h[i-1]/h[i]))
                if order != 2:
                    print('Order violated: {}'.format(order))
                    print('Error k,k+1: {}, {} \n'.format(e0, e1))

                    valid = False

        if valid:
            print('Gradient is valid')
            
'''Other'''
def logisticError(w, X, y):
    cls = np.rint(sps.expit(X@w))
    for i in range(cls.shape[0]):
        if cls[i] == 0:
            cls[i] = -1
            
    return np.sum(np.abs(cls-y)/2)/y.shape[0]

#Implicit to explicit
#B_func is linear function, N is input dimensionality
def imp2exp(B_func, N):
    I = np.eye(N)
    return B_func(I)
