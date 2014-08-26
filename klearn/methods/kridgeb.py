
from klearn.methods import BaseKernelEstimator
from sklearn.base import TransformerMixin, RegressorMixin
import scipy.linalg
import numpy as np

class kRidgeRegressionB(BaseKernelEstimator, TransformerMixin, RegressorMixin):
    r"""
    Perform ridge regression in the feature space

    .. math: min_p \sum_i (p_i^T X_i - y_i)^2

    Parameters
    ----------
    kernel : klearn.kernels.AbstractKernel instance
        kernel object for computing inner products
    eta : float, optional
        regularization strength
    regularize_beta : bool, optional
        Typically the regularization term is placed on the coefficients
        of the linear function in the feature space (`:math:p`). 
        A slight modification is to instead regularize the coefficients 
        of beta, which correspond to the coeffiencients of the expansion
        of p in terms of the training data points:
        
        .. math: p = \sum_i \beta_i \Phi(X_i)

    """
    def __init__(self, kernel, eta=1.0, regularize_beta=False):
        
        super(kRidgeRegressionB, self).__init__(kernel)

        self.eta = eta
        self.regularize_beta = regularize_beta

    def fit(self, X, y, gram_matrix=None):
        """
        Fit the kernel ridge regressor

        Parameters
        ----------
        X : np.ndarray, shape = [n_points, n_features]
            Indendent variables to fit. 
        y : np.ndarray, shape = [n_points]
            Dependent variables to fit to.
        gram_matrix : np.ndarray, optional, shape = [n_points, n_points]
            UNCENTERED gram matrix of inner products. It can be faster to pass a precomputed
            matrix if you're using it more than once. If None, we will compute it 
            again. 
        """

        self._Xtrain = X

        n_points = len(X)
        if gram_matrix is None:
            self.Ku = self.kernel(X)        

        else:
            if gram_matrix.shape != (n_points, n_points):
                raise Exception("Gram matrix is not the correct shape")

            self.Ku = gram_matrix

        self.Ku = np.vstack([self.Ku, np.ones(self.Ku.shape[1])])

        if self.regularize_beta:
            K2 = np.eye(n_points + 1)

        else:
            K2 = np.hstack([self.Ku, np.ones(n_points + 1).reshape((-1, 1))])
            K2[-1, :] = 0.0
            K2[:, -1] = 0.0
            K2[-1, -1] = 1.0

        a = self.Ku.dot(self.Ku.T) + self.eta * K2
        b = self.Ku.dot(y)

        sol = scipy.linalg.solve(a, b, sym_pos=False)

        self.beta = sol[:-1]
        self.y_intercept = np.float(sol[-1])

        #if self.regularize_beta:
        #    A = np.eye(n_points + 1)
        #    A[-1, -1] = 0.0
        #    mat = self.Ku.dot(self.Ku.T) + self.eta * A
        #    self.beta = np.linalg.inv(mat).dot(self.Ku.dot(y))
        #else:
        #    #A = np.vstack([np.eye(n_points), np.ones(n_points)])
        #    #mat = A.dot(self.Ku.T) + self.eta * np.ones(n_points + 1)
        #    #self.beta = np.linalg.inv(mat).dot(A.dot(y))
        #
        #    mat = self.Ku.dot(self.Ku.T) + self.eta * K2
        #    self.beta = np.linalg.inv(mat).dot(self.Ku.dot(y))
        #
        #self.beta = self.beta.reshape((-1, 1))[:-1]
        
        #self.y_intercept = np.float(self.beta[-1])

        return self

    def predict(self, X):
        """
        predict the dependent variable's value for new data

        Parameters
        ----------
        X : np.ndarray, shape = [n_points, n_features]
            Data for prediction. 
        
        Returns
        -------
        y_pred : np.ndarray, shape = [n_points]
            Predicted values.
        """

        Ku = self.kernel(self._Xtrain, X)
        n_points = len(self._Xtrain)

        y_pred = Ku.T.dot(self.beta) + self.y_intercept

        return y_pred.flatten()

    
    def transform(self, X):
        """
        transform the data onto its prediction

        Parameters
        ----------
        X : np.ndarray, shape = [n_points, n_features]
            Data for prediction.
    
        Returns
        -------
        y_pred : np.ndarray, shape = [n_points]
            Predicted values.
        """
        # this is the same as predict

        return self.predict(X)
