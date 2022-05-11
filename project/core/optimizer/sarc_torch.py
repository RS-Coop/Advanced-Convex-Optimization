'''
Custom PyTorch optimizer class that uses second order ARC.
'''
import torch
from torch.optim import Optimizer

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, eigsh

import time

'''
Helper functions
'''
'''Convert a list of tensors to a single 1D tensor'''
def list2vec(l):
    vec = [li.reshape(-1).data for li in l]

    return torch.cat(vec, 0)


'''Convert a 1D tensor to a list of tensors using template for the shapes'''
def vec2list(v, template):
    l = []
    start = 0
    for ti in template:
        l.append(v[start:start+torch.numel(ti)].reshape(ti.shape))
        start += torch.numel(ti)

    return l

'''Stochastic Adaptive Regularization with Cubics'''
class SARC(Optimizer):
    '''
    Initialize the optimizer.

    params -> Model parameters
    kw -> Optimizer keywords
    '''
    def __init__(self, params, sigma=1, eta_1=0.1, eta_2=0.9, gamma_1=2,
                    gamma_2=2, sub_prob_fails=1, sub_prob_max_iters=50,
                    sub_prob_tol=1e-6, eigen_tol=1e-3, sub_prob_method='explicit'):

        defaults = dict(sigma=sigma, eta_1=eta_1, eta_2=eta_2, gamma_1=gamma_1,
                        gamma_2=gamma_2, sub_prob_fails=sub_prob_fails,
                        sub_prob_max_iters=sub_prob_max_iters, sub_prob_tol=sub_prob_tol,
                        sub_prob_method=sub_prob_method, eigen_tol=eigen_tol)

        self.hvp_time, self.hvp_calls = 0.0, 0
        self.sub_time, self.sub_calls = 0.0, 0
        self.minimize_time, self.minimize_calls = 0.0, 0
        self.rho = []
        self.props = []

        if sub_prob_method == 'implicit':
            self._subProbSetup = self._implicitEig
        elif sub_prob_method == 'explicit':
            self._subProbSetup = self._explicitEig
        else:
            raise ValueError('Sub-problem solution method not supported.')

        super().__init__(params, defaults)


    '''
    Hessian vector product

    v -> The vector in question
    grads -> Gradients of the model
    '''
    def _hvp(self, v, gradsH):
        self.props[-1] += 1 #Add a propagation for the backward pass here
        self.hvp_calls += 1
        tic = time.perf_counter()

        v_list = vec2list(v, gradsH)

        hvp = torch.autograd.grad(gradsH, self.param_groups[0]['params'],
                                    grad_outputs=v_list, only_inputs=True,
                                    retain_graph=True)

        hvp_vec = list2vec(hvp)

        self.hvp_time += time.perf_counter()-tic

        return hvp_vec


    '''Evaluate cubic sub-problem and its gradient'''
    def _cubic(self, s, g, gH, sigma):
        Hs = self._hvp(s, gH)
        s_norm = torch.norm(s, 2)

        func = torch.dot(g, s) + 0.5*torch.dot(s, Hs) + (sigma/3)*s_norm.pow(3)
        grad = g + Hs + (sigma*s_norm)*s

        return func, grad


    '''Same as above but using Numpy'''
    def _cubic_np(self, s, g, H, sigma):
        func = g.T@s + 0.5*s.T@H@s + (sigma/3)*(np.linalg.norm(s)**3)
        grad = g + H@s + sigma*np.linalg.norm(s)*s

        return func, grad


    '''Solve the cubic sub-problem using generalized Lanczos'''
    def _implicitEig(self, grads, gradsH):
        self.sub_calls += 1
        tic = time.perf_counter()

        #Set device
        device = grads[0].device

        #Setup some parameters
        sigma = self.param_groups[0]['sigma']
        maxitr = self.param_groups[0]['sub_prob_max_iters']
        tol = self.param_groups[0]['sub_prob_tol']

        #Convert the gradients to a vector
        g = list2vec(grads)

        d = torch.numel(g)
        K = min(d, maxitr)
        Q = torch.zeros((d, K), device=device)

        q = g/torch.norm(g)

        T = torch.zeros((K, K), device=device)

        g_norm = torch.norm(g, 2)
        tol = min(tol, tol*g_norm)

        for i in range(K-1):
            Q[:,i] = q
            v = self._hvp(q, gradsH)
            T[i,i] = torch.dot(q, v)

            #Orthogonalize
            r = v - torch.matmul(Q[:,:i], torch.matmul(torch.transpose(Q[:,:i], 0, 1), v))

            b = torch.norm(r, 2)
            T[i,i+1] = b
            T[i+1,i] = b

            if b < tol:
                q = torch.zeros_like(q, device=device)
                break

            q = r/b

        #Compute last diagonal element
        T[i+1, i+1] = torch.dot(q, self._hvp(q, gradsH))

        T = T[:i+2, :i+2]
        Q = Q[:, :i+2]

        '''Now that we have split up the setup and solve this logic needs to change'''
        if torch.norm(T) < tol and g_norm < 1e-16:
            return torch.zeros(d, device=device), 0

        gt = torch.matmul(torch.transpose(Q, 0, 1), g)

        self.sub_time += time.perf_counter() - tic

        return (gt, T, Q, tol)

    '''Solve the cubic sub-problem using eigsh to get the eigen-point'''
    def _explicitEig(self, grads, gradsH):
        self.sub_calls += 1
        tic = time.perf_counter()

        #Set device
        device = grads[0].device

        #Setup some parameters
        sigma = self.param_groups[0]['sigma']
        maxitr = self.param_groups[0]['sub_prob_max_iters']
        tol = self.param_groups[0]['sub_prob_tol']
        eigen_tol = self.param_groups[0]['eigen_tol']

        #Convert the gradients to a vector
        g = list2vec(grads)
        d = torch.numel(g)

        #Define scipy linear operator
        def hvp(x):
            x = torch.from_numpy(x)
            return self._hvp(x, gradsH).numpy()

        hvp_lin_op = LinearOperator((d,d), matvec=hvp)

        #Calculate eigenvectors
        '''This can break sometimes and im not sure why.
        Need to make it a bit more robust'''
        try:
            _, evec = eigsh(hvp_lin_op, k=1, which='SA', maxiter=maxitr, tol=eigen_tol)
        except exception as e:
            evec = e.eigenvectors

        e = torch.from_numpy(evec.ravel())

        Q = torch.zeros((d, 2), device=device)
        T = torch.zeros((2, 2), device=device)

        g_norm = torch.norm(g, 2)
        tol = min(tol, tol*g_norm)
        q = g/g_norm

        Q[:,0] = q
        T[0,0] = g_norm
        a = torch.dot(q, e)
        T[0,1] = a
        T[1,0] = 0

        #Orthogonalize
        r = e - T[0,1]*q
        b = torch.norm(r, 2)
        T[1,1] = b

        if b < tol:
            q = torch.zeros_like(q, device=device)
        else:
            q = r/b
        Q[:,1] = q

        '''Now that we have split up the setup and solve this logic needs to change'''
        if torch.norm(T) < tol and g_norm < 1e-16:
            return torch.zeros(d, device=device), 0

        gt = torch.matmul(torch.transpose(Q, 0, 1), g)

        self.sub_time += time.perf_counter() - tic

        return (gt, T, Q, tol)


    def _subProbSolve(self, gt, T, Q, tol, sigma):
        self.minimize_calls += 1
        tic = time.perf_counter()

        z0 = np.zeros(T.shape[0])
        out = minimize(self._cubic_np, z0, args=(gt.numpy(), T.numpy(), sigma),
                        method='L-BFGS-B', tol=tol, jac=True)

        z = out['x']
        m = out['fun']
        s = torch.matmul(Q, torch.from_numpy(z).float())

        self.minimize_time += time.perf_counter() - tic

        return s, m


    '''Update model parameters inplace'''
    def _update(self, s):
        start = 0

        for p in self.param_groups[0]['params']:
            if p.requires_grad == True:
                n = torch.numel(p)
                p.data.add_(s[start:start+n].view(p.shape))

                start += n


    '''
    Step forward and update parameters using optimizer.

    grads -> Full list of gradients
    gradsH -> Partial list of gradients for sub-sampling Hessian
    loss -> Current loss value
    loss_fn -> Recomputes loss without retaining gradient information
    closure -> Clears gradients and re-evaluates model
    '''
    @torch.no_grad()
    def step(self, grads, gradsH, loss, loss_fn, closure=None):
        self.props.append(4) #4 for grads and gradsH

        sigma = self.param_groups[0]['sigma']
        eta_1 = self.param_groups[0]['eta_1']
        eta_2 = self.param_groups[0]['eta_2']
        gamma_1 = self.param_groups[0]['gamma_1']
        gamma_2 = self.param_groups[0]['gamma_2']
        sub_prob_fails = self.param_groups[0]['sub_prob_fails']

        fails = 0

        out = self._subProbSetup(grads, gradsH)

        while fails < sub_prob_fails:
            self.props[-1] += 1 #For calling loss_fn below
            s, m = self._subProbSolve(*out, sigma)

            #Need to figure out how its gonna work with updating parameters
            self._update(s)
            p = (loss - loss_fn())/(-m)
            self.rho.append(p.item())

            #bad update
            if p<eta_1:
                #Undo update
                self._update(-s)
                sigma = gamma_1*sigma
                fails += 1

            #great update
            elif p>=eta_2:
                sigma = max(sigma/gamma_2, 1e-16)
                break

            else:
                break

        #Update sigma
        self.param_groups[0]['sigma'] = sigma

    '''
    Get performance details
    '''
    def getInfo(self):
        return {
                'avg_hvp_time': self.hvp_time/self.hvp_calls,
                'avg_minimize_time': self.minimize_time/self.minimize_calls,
                'avg_sub_time': self.sub_time/self.sub_calls,
                'rho_list': self.rho,
                'props_list': self.props
        }
