
from klearn.methods import BaseKernelEstimator
from sklearn.base import TransformerMixin, ClassifierMixin
import numpy as np

class kLDA(BaseKernelEstimator, TransformerMixin, ClassifierMixin):
    r"""
    Perform kernel Linear Discriminant Analysis (which is the same as
    kernel Fisher's Linear Discriminant) for multiple classes

    The goal is to find successive projections that maximize the 
    between class variance, while minimizing the within class variance

    .. math: max_w \frac{w^T \Sigma_B w}{w^T \Sigma_W w}

    Parameters
    ----------
    kernel : klearn.kernels.AbstractKernel instance
        kernel object
    eta : float, optional
        regularization strength to apply
    """
    def __init__(self, kernel, eta=1.0):
        super(self, kLDA).__init__(kernel)

        self.eta = float(eta)


    def fit(self, X, y, gram_matrix=None):
        """
        Fit the LDA model with data
        
        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]
            training data
        y : np.ndarray, shape = [n_samples], type=int
            class assignments (should be contiguous integers
            starting from zero)
        gram_matrix : np.ndarray, optional, shape = [n_samples, n_samples]
        """

        n_points = len(X)
        if n_points != len(y):
            raise Exception("there should be one class label per data point")

        if np.any((y - y.astype(int)) > 1E-3):
            raise Exception("class labels should be integers")
        y = y.astype(int)
    
        uniq_y = np.unique(y)
        if (len(uniq_y) != (uniq_y[-1] + 1)) and (uniq_y[0] != 0):
            raise Exception("class labels should start at zero and be contiguous")
        self.n_classes = len(uniq_y)
        class_pops = np.bincount(y)

        if gram_matrix is None:
            self.Ku = self.kernel(X)

        else:
            if gram_matrix.shape != (n_points, n_points):
                raise Exception("gram matrix is not the correct shape")

            self.Ku = gram_matrix

        lhs = np.zeros((n_points, n_points))
        rhs = np.zeros((n_points, n_points))

        # speed note!
        # The outer products in this loop might be faster if we use
        # np.outer, but I don't know which is faster... Need to check
        for j in xrange(self.n_classes):
            Kj = self.Ku[:, np.where(y == j)[0]]
            Mj = Kj.mean(axis=1)

            A = Mj.dot(Mj.T)

            lhs += Kj.dot(Kj.T) * class_pops[j] - A
            rhs += A
        
        lhs /= float(self.n_classes)
        rhs /= float(self.n_classes)
        M = self.Ku.mean(axis=1)
        rhs += M.dot(M.T)

        # regularize
        rhs += np.eye(n_points) * self.eta 

        self.vals, self.betas = np.linalg.eigh(lhs, b=rhs)
        inc_ind = np.argsort(self.vals)[::-1]
        self.vals = self.vals[inc_ind]
        self.betas = self.betas[:, inc_ind]        

        self.projected_means = []
        # now compute where the projected means end up
        for j in xrange(self.n_classes):
            Kj = self.Ku[:, np.where(y == j)[0]]
            Mj = Kj.mean(axis=1, keepdims=True)

            self.projected_means.append(Mj.T.dot(self.betas[:, :(self.n_classes - 1)]))

        self.projected_means = np.array(self.projected_means)

        return self


    def transform(self, X):
        """
        transform new data into the `n_classes - 1` dimensional
        space for classification

        Parameters
        ----------
        X : np.ndarray, shape = [n_points, n_features]
            data to transform

        Returns
        -------
        Xnew : np.ndarray, shape = [n_points, n_classes - 1]
        """

        Ku = self.kernel(self._Xtrain, X)

        Xnew = Ku.T.dot(self.betas[:, :(self.n_classes - 1)])

        return Xnew


    def predict(self, X):
        """
        classify new data into one of the classes
        
        Parameters
        ----------
        X : np.ndarray, shape = [n_points, n_features]
            new data to classify

        Returns
        -------
        y_pred : np.ndarray, shape = [n_points]
            predicted classification
        """

        Xnew = self.transform(X)

        # now, it's just an assignment thing.
        y_pred = np.ones(Xnew.shape[0]) - 1
        for i in xrange(Xnew.shape[0]):
            y_pred[i] = np.argmin(np.square(Xnew[i] - self.projected_means).sum(1))

        return y_pred

        
