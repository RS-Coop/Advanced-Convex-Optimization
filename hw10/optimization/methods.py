'''
A variety of optimization algorithms
'''
from warnings import warn

import numpy as np
import numpy.linalg as la

'''
Implements linesearch procedure for gradient based stepsize determination

Input:
    x -> current location
    p -> search direction
    t -> initial stepsize
    f -> function
    grad -> gradient
Output:
    t -> stepsize in search direction p
'''
def linesearch(x, p, t, f, grad, args, c=1e-4, r=0.9):

    fk = f(x, *args)
    gk = grad(x, *args)

    while f(x+t*p, *args) > \
            fk + c*t*(gk.T@p):

        t = r*t

    return t

'''
Implements the standard dradient descent method for optimization

Input:
    x0 -> Starting location
    f -> Function
    grad -> Function gradient
    stepsize -> Initial step-size
    args -> Additional arguments for f and grad (optional)
    steptype -> Fixed or linesearch (optional)
    rtol -> Relative tolerance (optional)
    atol -> Absolute tolerance (optional)
    maxItr -> Maximum iterations (optional)
    return_iterates -> Return sequence of iterates (optional)
Output:
    - Minimizer of f
    - Sequence of iterates (optional)
'''
def gradDescent(x0, f, grad, step, args=(), steptype='fixed',
                rtol=1e-5, atol=1e-8, maxItr=1000, return_iterates = False):

    if steptype not in ['fixed', 'linesearch']:
        raise ValueError('Argument steptype must be one of "fixed" or "linesearch"')

    x_vec = [x0]

    for i in range(maxItr):
        x1 = x0 - step*grad(x0, *args)

        x_vec.append(x1)

        if la.norm(x1-x0) < atol + rtol*la.norm(x0):
            return x1, np.array(x_vec)
        
        if steptype == 'linesearch':
            step = linesearch(x1, -grad(x1, *args), step*2, f, grad, args)

        x0 = x1

    warn('Maximum iterations exceeded without \
                  reaching specified tolerance.', RuntimeWarning)

    return x1, np.array(x_vec)

'''
Implements proximal Nesterov accelerated gradient descent. This formulation
also allows for standard Nesterov gradient descent when the proximity operator
is the identity.

Input:
    x0 -> Starting location
    f -> Function
    grad -> Function gradient
    prox -> Proximity operator
    stepsize -> Initial step-size
    args -> Additional arguments for f and grad (optional)
    prox_args -> Additional arguments for prox (optional)
    steptype -> Fixed or linesearch (optional)
    rtol -> Relative tolerance (optional)
    atol -> Absolute tolerance (optional)
    maxItr -> Maximum iterations (optional)
Output:
    - Minimizer
    - Sequence of iterates (optional
'''
def proximalNGD(x0, f, grad, prox, step, args=(), prox_args=(), steptype='fixed',
                rtol=1e-5, atol=1e-8, maxItr=1000, ret_iterates=False):
    
    #Check on steptype
    if steptype not in ['fixed', 'linesearch']:
        raise ValueError('Argument steptype must be one of "fixed" or "linesearch"')
        
    #Check on proximity operator
    if prox == None:
        prox = lambda y: y #Identity
    elif not callable(prox):
        raise ValueError('Proximity operator must be callable function or None.')

    #Okay now we are ready       
    x_vec = [x0] #First iterate
    
    #Initial y and t
    y0 = x0
    t0 = 1

    for i in range(maxItr):
        #NOTE: I was also passing in step, but that doesn't work with all prox
        x1 = prox(y0 - step*grad(y0, *args), *prox_args) #Compute next step

        x_vec.append(x1) #Save iterates

        #Check for convergence
        if la.norm(x1-x0) < atol + rtol*la.norm(x0):
            if ret_iterates:
                return x1, np.array(x_vec)
            else: return x1
        
#         #Compute new step-size
#         if steptype == 'linesearch':
#             step2 = methods.linesearch(y0, x1-y0, 1, f, grad, args)
#             x1 = y0 + step2*(x1-y0)
#             step = step2*step*1.1
            
        #Update
        t1 = (1 + np.sqrt(1+4*t0**2))/2
        y0 = x1 + ((t0-1)/t1)*(x1 - x0)
            
        #Reset for next iteration
        t0 = t1
        x0 = x1

    warn('Maximum iterations exceeded without \
                  reaching specified tolerance.', RuntimeWarning)

    if ret_iterates:
        return x1, np.array(x_vec)
    else: return x1