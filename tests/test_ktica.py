
from sklearn.grid_search import GridSearchCV
from klearn.methods import ktICA
from klearn.kernels import Linear, RBF
import numpy as np

def test_ktica():
    # this is a really easy test, since the classes are 
    # generated from two different MVN's
    

    rbf_kernel = RBF()

    tica_model = ktICA(rbf_kernel, eta=1.0)

    param_set = {'eta': [1E-4, 1E-2, 1, 1E2, 1E4],
                 'kernel__sigma': [0.01, 0.05, 0.1, 0.25, 0.5]}

    gscv = GridSearchCV(lda_model, param_set)  
    gscv.fit(X, y)

    lda_model = gscv.best_estimator_
    print gscv.best_params_
    

if __name__ == '__main__':
    test_ktica()
