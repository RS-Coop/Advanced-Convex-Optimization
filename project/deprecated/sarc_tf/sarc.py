'''
Custom TensorFlow Optimizer that implements ARC
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizer_v2.optimizer_v2 import OptimizerV2

import hessian as H
import sub_problem as cubic

'''Stochastic Adaptive Regularization with Cubics'''
class SARC(OptimizerV2):
    '''
    Initialize the optimizer.
    '''
    def __init__(self, name='SARC_Optimizer', sigma=1, eta_1=0.1, eta_2=0.9,
                    gamma_1=2, gamma_2=2, sub_problem_fails=10, **kwargs):

        super().__init__(name, **kwargs)
        self._set_hyper('sigma_base', sigma)
        self._set_hyper('eta_1', eta_1)
        self._set_hyper('eta_2', eta_2)
        self._set_hyper('gamma_1', gamma_1)
        self._set_hyper('gamma_2', gamma_2)

        self.sigma = sigma
        self.sub_problem_fails = sub_problem_fails

    '''Get optimizer configuration details.'''
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            'sigma_base': self._serialize_hyperparameter('sigma_base'),
            'eta_1': self._serialize_hyperparameter('eta_1'),
            'eta_2': self._serialize_hyperparameter('eta_2'),
            'gamma_1': self._serialize_hyperparameter('gamma_1'),
            'gamma_2': self._serialize_hyperparameter('gamma_2')
        }

    '''Not Implemented.'''
    # def _create_slots(self, vars):
    #     pass

    '''
    This is just a hacky method to get the hvp and closure functions which we
    need in _resource_apply_dense. Call this before calling apply_gradients.

    closure -> Recompute loss with updated parameters
    hess -> Function to compute model hessian vector product
    '''
    def prepare(self, loss, closure, hess):
        self.loss = loss
        self.closure = closure
        self.hess = hess

    '''
    Main functionality to update parameters of model.

    grad -> Model gradient
    vars -> Model parameters
    '''
    @tf.function
    def _resource_apply_dense(self, grad, vars):
        #Get the hyperparameters
        var_dtype = vars.dtype.base_dtype

        eta_1 = self._get_hyper('eta_1', var_dtype)
        eta_2 = self._get_hyper('eta_2', var_dtype)
        gamma_1 = self._get_hyper('gamma_1', var_dtype)
        gamma_2 = self._get_hyper('gamma_2', var_dtype)

        #Failure counter
        fails = 0

        '''
        Right now we are just trying to update a single time, and if it fails
        then we return anyways without an update.

        We could also loop until we achieve an update.
        '''
        #Solve sub-problem
        update, m = cubic(grad, self.hess, self.sigma)

        #Evaluate update
        p = (loss - self.closure(update))

        #Update vars and trust-region
        #Good step
        if p>=self._get_hyper('eta_1'):
            vars += update

        #Okay step
        elif p>=self._get_hyper('eta_2'):
            self.sigma = max(self.sigma/gamma_2, 1e-16)

        #Bad step
        else:
            self.sigma = gamma_1*self.sigma

            fails += 1
            print('Failed to update on current step.')

    '''Not Implemented.'''
    @tf.function
    def _resource_apply_sparse(self):
        raise NotImplementedError
