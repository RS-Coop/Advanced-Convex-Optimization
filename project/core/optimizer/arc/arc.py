'''
Adaptive Regularization with Cubics (ARC) implementation.

This implementation is meant to be general, but we expect the possibility
of using approximate hessian information and only approximately solving the
sub-problem.

Algorithms for solving the sub-problem are given in sub_problem.py
'''

from warnings import warn
import numpy as np
import numpy.linalg as la

from .sub_problem import arcSub

'''
Approximate Adaptive Regularization with Cubics

Input:
    x0 -> Starting point
    F -> Objective function
    gradF -> Gradient of F (callable)
    hessF -> Hessian of F (callable)
    args -> Additional arguments for function, gradient, and hessian (optional)
    eps_g -> Gradient tolerance (optional)
    eps_h -> Hessian tolerance (optional)
    sigma -> Initial regularization parameter (0,inf) (optional)
    eta -> Step success (0,1] (optional)
    gamma -> Regularization update (1,inf) (optional)
    maxitr -> Maximum number of iterations (optional)
    sub_method -> Sub-problem solver (optional)
    sub_tol -> Sub-problem tolerance (optional)
    sub_maxitr -> Maximum sub-problem solver iterations (optional)
Output:
    x -> Minimizer

NOTE: Look into stopping condtions, code seems to be different than paper
algorithm in sources.
'''
def arc(x0, F, gradF, hessF, args=(), eps_g=1e-3, eps_h=1e-3, sigma=1, eta_1=0.1,
        eta_2=0.9, gamma_1=2, gamma_2=2, maxitr=1000, sub_method='lanczos'):

    fails = 0 #Keep track of failed updates

    xt = x0

    #Set current objective value, gradient, and hessian
    ft = F(xt, *args)
    gt = gradF(xt, *args)
    Ht = hessF(xt, *args)

    #Check termination conditions
    #Bounds on norm of Gradient and smallest eigenvalue of hessian
    if la.norm(gt)<=eps_g:
        gt = np.zeros(gt.shape)
        if la.eigvals(Ht).min()>=-eps_h:
            return xt

    for i in range(maxitr):
        #Solve sub-problem
        #Get step (s) and objective value at s (m)
        s, m = arcSub(sub_method, (gt, Ht, sigma))

        #Evaluate how good our step was
        p = (ft - F(xt+s))/(-m)

        #If step was good update
        if p>=eta_1:
            xt = xt + s

            #Update gradient and hessian accordingly
            ft = F(xt, *args)
            gt = gradF(xt, *args)
            Ht = hessF(xt, *args)

            #Check termination conditions
            #Bounds on norm of Gradient and smallest eigenvalue of hessian
            if la.norm(gt)<=eps_g:
                gt = np.zeros(gt.shape)
                if la.eigvals(Ht).min()>=-eps_h:
                    return xt

        #Okay update
        if p>=eta_2:
            sigma = max(sigma/gamma_2, 1e-16)

        #Bad update
        elif p<eta_1:
            sigma = gamma_1*sigma

            fails += 1
            if fails == 50:
                print('Failure, exiting.')
                return xt

    warn('WARNING! Maximum iterations exceeded \
            without achieving tolerance.', RuntimeWarning)

    return xt
