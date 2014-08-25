
import numpy as np
import scipy.linalg
from mdtraj import io
from sklearn.base import TransformerMixin
from klearn.estimators import BaseKernelEstimator

class kPCA(BaseKernelEstimator, TransformerMixin):
    """ 
    class for calculating PCs in a high dimensional feature space
    """

    def __init__(self, kernel, n_components=1):

        """
        Initialize an instance of the kPCA solver

        Paramaters
        ----------
        kernel : klearn.kernels.AbstractKernel instance
            kernel object
        n_components : int, optional
            number of PC's to use to transform the data
        """
        super(self, kPCA).__init__(kernel)

        self._Xall = None
        self.K = None


    def fit(self, X, gram_matrix=None):
        """
        Fit the kPCA model to data, X. If gram_matrix is not None, then
        a gram matrix will not be computed for the data, and this matrix
        will be used.

        Parameters
        ----------
        X : np.ndarray
            data to fit (n_points, n_features)
        gram_matrix : np.ndarray, optional
            precomputed UNCENTERED gram matrix (this is faster if you're fitting
            many models) (n_points, n_points). 
        """
        n_points = len(X)
        self._Xtrain = X

        if not gram_matrix is None:
            self.Ku = self.kernel(X)

            oneN = np.ones((n_points, n_points)) / float(n_points)
            A = oneN.dot(self.K)
            self.K = self.Ku - 2 * A + A.dot(oneN)

        else:
            self.Ku = gram_matrix

            if gram_matrix.shape != (n_points, n_points):
                raise Exception("gram matrix is not the correct shape")

            oneN = np.ones((n_points, n_points)) / float(n_points)
            A = oneN.dot(self.K)
            self.K = self.Ku - 2 * A + A.dot(oneN)

        # in klearn, the solution vectors are always called "beta"
        self.vals, self.betas = np.linalg.eigh(self.K / float(n_points))
        inc_ind = np.argsort(self.vals)[::-1]
        # note that sometime in the future we might want the negative
        # eigenvalues (or at least for certain kernels, e.g. SCISSORS)
        self.vals = self.vals[inc_ind]
        self.betas = self.betas[:, inc_ind]

        return self

    def transform(self, X):
        """
        Transform new data onto the top `n_components` PC's
        
        Parameters
        ----------
        X : np.ndarray
            test data, with points in the rows. Shape should be 
            (n_points, n_features)

        Returns
        -------
        X_new : np.ndarray
            transformed data. The shape is (n_points, n_components)
        """
        Ku = self.kernel(self._Xtrain, X)
        
        n_train = len(self._Xtrain)
        oneN = np.ones((n_train, n_train)) / float(n_train)

        A = self.Ku.dot(oneN)
        K = Ku - oneN.dot(Ku) - A + oneN.dot(A)

        X_new = K.T.dot(self.betas[:, :self.n_components])

        return X_new
