
from sklearn.grid_search import GridSearchCV
from klearn.methods import kLDA
from klearn.kernels import Linear
import numpy as np

def test_klda():
    # this is a really easy test, since the classes are 
    # generated from two different MVN's
    
    dim = 5
    n = 100
    eta = 10
    sigma = 0.2
    
    mu0 = np.random.random(size=dim)
    vec = np.random.random(size=dim)
    vec /= np.sqrt(vec.dot(vec))
    mu1 = mu0 + vec

    X0 = np.random.normal(0, scale=sigma, size=(n, dim)) + mu0
    X1 = np.random.normal(0, scale=sigma, size=(n, dim)) + mu1

    X = np.concatenate([X0, X1])
    y = np.concatenate([np.zeros(n), np.ones(n)])

    linear_kernel = Linear()

    lda_model = kLDA(linear_kernel, eta=1.0)

    gscv = GridSearchCV(lda_model, [{'eta' : [1E-2, 1E-1, 1, 10, 100]}])  
    gscv.fit(X, y)

    lda_model = gscv.best_estimator_
    
    n_test = 100
    threshold = int(0.02 * 2 * n_test)
    
    Xtest0 = np.random.normal(0, scale=sigma, size=(n_test, dim)) + mu0
    Xtest1 = np.random.normal(0, scale=sigma, size=(n_test, dim)) + mu1

    ytest0 = lda_model.predict(Xtest0)
    ytest1 = lda_model.predict(Xtest1)

    n_wrong = len(np.where(ytest0 == 1)[0]) + len(np.where(ytest1 == 0)[0])

    print "got %d / %d wrong" % (n_wrong, 2 * n_test)

    assert n_wrong < threshold

if __name__ == '__main__':
    test_klda()
