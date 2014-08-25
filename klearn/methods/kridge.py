
from klearn.methods import BaseKernelEstimator
from sklearn.base import TransformerMixin, RegressorMixin
import numpy as np

class kRidgeRegression(BaseKernelEstimator, TransformerMixin, RegressorMixin):
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
        
        super(self, kRidgeRegression).__init__(kernel)

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

        self._ymean = y.mean()
        self._Xtrain = X

        n_points = len(X)
        if gram_matrix is None:
            self.Ku = self.kernel(X)        

        else:
            if gram_matrix.shape != (n_points, n_points):
                raise Exception("Gram matrix is not the correct shape")

            self.Ku = gram_matrix

        oneN = np.ones((n_points, n_points)) / float(n_points)

        A = oneN.dot(self.Ku)
        self.K = self.Ku - 2 * A + A.dot(oneN)

        if self.regularize_beta:
            mat = self.K.dot(self.K) + self.eta * np.eye(n)
            self.beta = np.linalg.inv(mat).dot(self.K.dot(y - self.ymean))
        else:
            mat = self.K + self.eta * np.eye(n)
            self.beta = np.linalg.inv(mat).dot(y - self.ymean)
        
        self.beta = self.beta.reshape((-1, 1))
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

        oneN = np.ones((n_points, n_points)) / float(n_points)
        A = self.Ku.dot(oneN)
        K = Ku - oneN.dot(Ku) - A + oneN.dot(A)

        # hey whoever's reading this, is this the correct thing if y wasn't
        # mean centered to begin with?
        y_pred = K.T.dot(self.beta) + self.ymean

        return y_pred

    
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
