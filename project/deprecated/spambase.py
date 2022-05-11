import pandas as pd
import numpy as np

#returns scalar
def _func(w, X, y):
    l = np.logaddexp(np.zeros(y.shape), -y*np.dot(X,w))
    return np.sum(l)

#returns vector
def _grad(w, X, y):
    u = sps.expit(-y*np.dot(X,w))

    return np.dot(-X, (y*u))

#returns matrix
def _hess(w, X, y):
    u = sps.expit(-y*np.dot(X,w))
    S = np.diag(u*(1-u))

    return X.T@S@X

def _logisticError(w, X, y):
    cls = np.rint(sps.expit(np.dot(X,w)))
    for i in range(cls.shape[0]):
        if cls[i] == 0:
            cls[i] = -1

    return np.sum(np.abs(cls-y)/2)/y.shape[0]

def spambase(method, kwargs={}):
    data = pd.read_csv('./spambase/spambase.data')

    X = data.iloc[:,0:-2]
    y = data.iloc[:,-1]

    X = X.apply(lambda x: np.log(x+0.1))
    y = y.apply(lambda x: 1 if x else -1)

    train_test_split(X, y)

    args_train = (X_train.to_numpy(), y_train.to_numpy())
    args_test = (X_test.to_numpy(), y_test.to_numpy())

    x0 = np.ones(args_train[0].shape[1])

    x = method(x0, func, grad, hess, args=args_train, **kwargs)

    return error(x, *args_test)

if __name__=='__main__':
    pass
