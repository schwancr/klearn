import numpy as np
import inspect
from sklearn.base import BaseEstimator

# inherit from BaseEstimator so that we can use the [sg]et_params methods
# this isn't really the best way to do it, because the kernel simply
# isn't an estimator... but this is useful. I could alternatively
# just write the [sg]et_params methods correctly but that seems a bit
# silly if I can just reuse sklearn's stuff 
class AbstractKernel(BaseEstimator):
    """Abstract class for all new kernel functions"""
    def __call__(self, X, Xtest=None):
        """
        Compute a gram matrix from given data.
        
        Parameters
        ----------
        X : array_like
            Array containing points in rows, and features in
            the columns. (shape: (n_points, n_features)). If Xtest 
            is None, then the returned gram matrix will consist of
            the inner products between all pairs in this array

            .. math: K[i, j] = <X[i], X[j]>

        Xtest : np.ndarray, optional
            If not none, then the gram matrix will be the inner product
            between points in X (in the rows) to the points in Xtest
            
            .. math: K[i, j] = <X[i], Xtest[j]>

        Notes
        -----
        X and Xtest need not be np.ndarray, the only requirement is that
        they have the __len__ and __getitem__ methods.


        Returns
        -------
        K : np.ndarray
            Gram matrix of inner products. NOTE: This matrix is NOT
            centered in feature space.
        """

        if Xtest is None:
            X2 = X
        else:
            X2 = Xtest

        n_rows = len(X)
        n_cols = len(X2)

        # potential speed-up: since K is symmetric when X2=X, we're
        # doing twice the work
        K = np.zeros((n_rows, n_cols))
        for i in xrange(n_rows):
            K[i] = self._kernel_function(X[i], X2)

        return K


    def _kernel_function(self, one, many):
        raise NotImplementedError()


    #def set_params(self, **kwargs):
    #    for key, value in kwargs.iteritems():
    #        if 
    #        setattr(self, key, value)
    #   
    #
    #def get_params(self, deep=True):
    #    """
    #    Get the parameters for the kernel. 
    #
    #    Returns
    #    -------
    #    params : dict 
    #        parameter names mapped to their values
    #
    #    Notes
    #    -----
    #    This behaves just as sklearn.base.BaseEstimator.get_params
    #    
    #    """
    #    out = dict()
    #    # Adopted from sklearn's BaseEstimator._get_param_names
    #    init = getattr(self.__class__, '__init__')
    #    args, varargs, keywards, defaults = inspect.getargspec(init)
    #
    #    args.pop(0) # remove 'self'
    #
    #    for key in args:
    #        out[key] = getattr(self, key)
    #
    #        # do what sklearn expects so that GridSearch works
    #        if deep:
    #            if hasattr(out[key], 'get_params'):
    #                deep_items = out[key].get_params(deep=True).items()
    #                out.update([(key + '__' + deep_key, deep_val) for deep_key, deep_val in deep_items])
    #                
    #    return out
