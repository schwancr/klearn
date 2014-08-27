
import numpy as np
import scipy.linalg
from mdtraj import io
import pickle
from klearn.methods import BaseKernelEstimator
from sklearn.base import TransformerMixin

class ktICA(BaseKernelEstimator, TransformerMixin):
    """ 
    class for calculating tICs in a high dimensional feature space
    """

    def __init__(self, kernel, dt, n_components=1, eta=1.0):

        """
        Initialize an instance of the ktICA solver

        Paramaters
        ----------
        kernel : klearn.kernels.AbstractKernel instance
            kernel object
        dt : int
            correlation lag time to compute the time-lag correlation matrix
        n_components : int, optional
            number of ktICs to use in transforming the data.
        eta : float, optional
            regularization strength
        """

        super(ktICA, self).__init__(kernel)

        self.eta = float(eta)
        self.dt = int(dt)
        self.n_components = int(n_components)


    def fit(self, X, X_dt=None, gram_matrix=None):
        r"""
        Fit the model to a given timeseries
        
        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]
            if X_dt is None, then this is a time series and pairs of 
            points will be sampled from it. Otherwise, these are the
            initial points to the corresponding points in X_dt
        X_dt : np.ndarray, optional, shape = [n_samples, n_features]
            If not none, then these are the points that are `dt` timesteps
            after the corresponding points in X
        gram_matrix : np.ndarray, optional, shape = [n_points * 2, n_points * 2]
            UNCENTERED gram matrix of inner products such that:
            
                `Xnew = concatenate([X, X_dt])`
                `gram_matrix[i, j] = inner(Xnew[i], Xnew[j])`
            
            The number of points varies depending on X_dt:
                - If X_dt is None, then `n_points = n_samples - dt`
                - If X_dt is an array, then `n_points = n_samples`
        """

        if X_dt is None:
            self._Xtrain = np.concatenate([X[:- self.dt], X[self.dt:]])
        else:
            self._Xtrain = np.concatenate([X, X_dt])

        n_points = len(self._Xtrain) / 2

        if gram_matrix is None:
            self.Ku = self.kernel(self._Xtrain)

        else:
            if gram_matrix.shape != (2 * n_points, 2 * n_points):
                raise Exception("gram matrix is not the correct shape")

            self.Ku = gram_matrix


        self.Ku = (self.Ku + self.Ku.T) * 0.5

        oneN = np.ones(self.Ku.shape[0]) / float(2 * n_points)
        oneN.reshape((-1, 1))

        self.K = self.Ku - oneN.T.dot(self.Ku) - self.Ku.dot(oneN) + oneN.T.dot(self.Ku.dot(oneN))

        self.K = (self.K + self.K.T) * 0.5

        R = np.zeros(self.K.shape)
        R[:n_points, n_points:] = np.eye(n_points)
        R[n_points:, :n_points] = np.eye(n_points)

        KK = self.K.dot(self.K)

        lhs = self.K.dot(R).dot(self.K)
        rhs = KK + self.eta * np.eye(2 * n_points)

        self.vals, self.betas = scipy.linalg.eigh(lhs, b=rhs)

        dec_ind = np.argsort(self.vals)[::-1]
        
        self.vals = self.vals[dec_ind]
        self.betas = self.betas[:, dec_ind]

        M = float(self.K.shape[0])

        vKK = self.betas.T.dot(KK)

        # not sure if I should compute the variance based on
        # the regularization strength or not :/
        self.vec_vars = np.sum(vKK * self.betas.T, axis=1) / M

        self.betas = self.betas / np.sqrt(self.vec_vars)


    def transform(self, X):
        """
        project a point onto the top `n_components` ktICs

        Parameters
        ----------
        X : np.ndarray, shape = [n_points, n_features]
            data to project onto eigenvector
        
        Returns
        -------
        Xnew : np.ndarray, shape = [n_points, n_components]
            projected value of each point in the trajectory
        """

        Ku = self.kernel(self._Xtrain, X)

        N = Ku.shape[0]
        oneN = np.ones((N, 1)) / float(N)

        K = Ku - self.Ku.dot(oneN) - oneN.T.dot(Ku) + oneN.T.dot(self.Ku.dot(oneN))

        Xnew = K.T.dot(self.betas[:, :self.n_components])
        
        return Xnew


    def score(self, X, X_dt=None, timestep=None):
        """
        Evaluate the solutions based on new data X. This uses the assumption that
        the ktICA solutions are the eigenfunctions of a Transfer operator.
        This means we can decompose the transfer operator into a sum of terms
        along each ktICA solution, which gives a probability of a new trajectory.

        Using Bayes' rule this can be translated into the likelihood of the
        ktICA solution.

        Parameters
        ----------
        X : np.ndarray
            data with features in the second axis
        X_dt : np.ndarray
            data such that X_dt[i] is observed one timestep after X[i]. Note
            that this timestep need not be the same as the dt specified for
            solving the ktICA solution
        timestep : time separating X from X_dt (should be in the same units
            as self.dt

        Returns
        -------
        log_like : float
            log likelihood of the new data given the ktICA solutions
        """

        if self.betas is None:
            return - np.inf

        if timestep is None:
            timestep = self.dt

        if timestep < self.dt:
            raise Exception("can't model dynamics less than original dt.")

        elif timestep == self.dt:
            exponent = 1

        else:
            if (timestep % self.dt):
                raise Exception("for timestep > dt, timestep must be a multiple of dt.")

            exponent = int(round(timestep / self.dt))

        if X_dt is None:
            X_dt = X[timestep:]
            X = X[:-timestep]

        proj_X = self.transform(X)
        proj_X_dt = self.transform(X_dt)
        
        N = proj_X.shape[0]
        # the first right eigenvector sends everything to unity
        proj_X = np.hstack([np.ones((N, 1)), proj_X])
        proj_X_dt = np.hstack([np.ones((N, 1)), proj_X_dt])

        vals = np.concatenate([[1], self.vals[:self.n_components]]).real
        vals = np.power(vals, exponent)
        vals = np.reshape(vals, (-1, 1))

        temp_array = proj_X * proj_X_dt
        # don't multiply by muA because that is the normalization
        # constraint on the output PDF
        temp_array = temp_array.dot(vals)
        # NOTE: The above likelihood is merely proportional to the actual likelihood
        # we would really need to multiply by a volume of phase space, since this
        # is the likelihood PDF...
        #temp_array[np.where(temp_array <=0)] = temp_array[np.where(temp_array > 0)].min() / 1E6
        bad_ind = np.where(temp_array <= 0)[0]
        if len(bad_ind) > 0:
            log_like = - np.inf

        else:
            log_like = np.log(temp_array).sum()

        return log_like


    def save(self, output_fn):
        """
        save results to a .h5 file
        """
    
        kernel_str = pickle.dumps(self.kernel)

        io.saveh(output_fn, vals=self.vals,
            betas=self.betas, K=self.K, 
            Ku=self.Ku, eta=np.array([self.eta]),
            Xtrain=self._Xtrain, dt=np.array([self.dt]),
            kernel_str=np.array([kernel_str]))


    @classmethod
    def load(cls, input_fn, kernel=None):
        """
        load a ktica object saved via the .save method. 

        Parameters
        ----------
        input_fn : str
            input filename
        kernel : kernel instance, optional
            kernel to use when calculating inner products. If None,
            then we will look in the file. If it's not there, then an 
            exception will be raised

        Returns
        -------
        kt : ktica instance
        """

        f = io.loadh(input_fn)

        if not kernel is None:
            kernel = kernel
        elif 'kernel_str' in f.keys():
            kernel = pickle.loads(f['kernel_str'][0])
        else:
            raise Exception("kernel_str not found in %s. Need to pass a kernel object")

        kt = cls(kernel, f['dt'][0], eta=f['eta'][0])  
        # dt and reg_factor were saved as arrays with one element

        kt.Ku = f['Ku']
        kt.K = f['K']

        kt._Xtrain = f['Xtrain'].astype(np.double)

        kt.vals = f['vals']
        kt.betas = f['betas']
     
        return kt

