import numpy as np
from klearn.methods import kRidgeRegressionB as kRidgeRegression
from klearn.kernels import Linear
from sklearn.metrics import r2_score
from matplotlib.pyplot import *
from sklearn.grid_search import GridSearchCV

def test_krr_regP():
    
    dim = 5
    n = 1000
    ntest = 1001    

    pref = np.random.random(size=dim) - 0.5

    #pref /= np.sqrt(pref.dot(pref))

    Xtrain = np.random.random((n, dim)) + 1.0
    ytrain = Xtrain.dot(pref) + np.random.normal(scale=0.05, size=n) + 10.0

    Xtest = np.random.random((ntest, dim)) + 1.0
    yref = Xtest.dot(pref) + 10.0

    krr = kRidgeRegression(kernel=Linear(), eta=1.0)

    gs = GridSearchCV(krr, {'eta' : [0, 1E-16, 1E-14, 1E-12, 1E-10, 1E-8, 1E-6, 1E-4, 1E-2, 1]})

    gs.fit(Xtrain, ytrain)

    krr = gs.best_estimator_

    ytest = krr.transform(Xtest).flatten()
    print krr.beta.shape
    print krr.Ku.shape

    print krr.score(Xtest, yref)
    #assert krr.score(Xtest, yref) > 0.98

    #scatter(Xtrain.dot(pref), ytrain, color='blue', alpha=0.5)
    #scatter(yref, ytest, color='red', alpha=0.5)
    #show()

def test_krr_regbeta():
    
    dim = 5
    n = 1000
    ntest = 1001    

    pref = np.random.random(size=dim) - 0.5

    #pref /= np.sqrt(pref.dot(pref))

    Xtrain = np.random.random((n, dim)) 
    ytrain = Xtrain.dot(pref) + np.random.normal(scale=0.05, size=n) + 10.0

    Xtest = np.random.random((ntest, dim)) 
    yref = Xtest.dot(pref) + 10.0

    krr = kRidgeRegression(kernel=Linear(), eta=1.0, regularize_beta=True)
    gs = GridSearchCV(krr, {'eta' : [1E-6, 1E-4, 1E-2, 1, 1E2, 1E4, 1E6]})
    gs.fit(Xtrain, ytrain)

    krr = gs.best_estimator_

    ytest = krr.transform(Xtest).flatten()

    print krr.score(Xtest, yref)
    #assert krr.score(Xtest, yref) > 0.98

    #scatter(Xtrain.dot(pref), ytrain, color='blue', alpha=0.5)
    #scatter(yref, ytest, color='red', alpha=0.5)
    #show()

if __name__ == '__main__':
    test_krr_regP()
    test_krr_regbeta()
