
from sklearn.grid_search import GridSearchCV
from klearn.methods import ktICA
from klearn.kernels import Linear, RBF
from matplotlib.pyplot import *
import numpy as np

def test_ktica():
    T = np.array([[0.98, 0.02], [0.01, 0.99]])
    vals, vecs = np.linalg.eig(T.T)
    ind = np.argsort(vals)[::-1]
    vals = vals[ind]
    vecs = vecs[:, ind]

    states = [0]
    for i in xrange(10000):
        states.append(np.argmax(np.random.multinomial(1, T[states[-1]])))

    states = np.array(states)
    noise = np.random.normal(scale=0.25, size=len(states))

    mus = np.array([0, 5])

    traj = mus[states] + noise

    traj = traj.reshape((-1, 1))

    print traj.min(), traj.max()

    dt = 10

    X = traj[:-dt]
    X_dt = traj[dt:]    


    X = X[::50]
    X_dt = X_dt[::50]

    rbf_kernel = RBF()
    linear_kernel = Linear()

    tica_model = ktICA(rbf_kernel, dt=dt, eta=2.5, n_components=1)

    param_set = {'eta': [1E-8, 1E-6, 1E-4, 1E-2, 1, 100],
                 'kernel__sigma' : [0.25, 0.5, 1, 2]}

    gscv = GridSearchCV(tica_model, param_set)  
    gscv.fit(X, X_dt)

    x = np.linspace(-5, 10, 100).reshape((-1, 1))
    y = gscv.transform(x)

    lam_est = gscv.best_estimator_.vals[0]
    lam_ref = vals[1] ** dt

    print gscv.best_params_
    print - dt / np.log(lam_ref), - dt / np.log(lam_est)
    
    plot(x, y, lw=3)
    show()

if __name__ == '__main__':
    test_ktica()
