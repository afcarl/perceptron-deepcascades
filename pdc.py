from __future__ import division

import perceptron
import csv
import numpy as np
from numpy import linalg
import cPickle as pickle
import copy

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


class PerceptronDeepCascade(object):
    def __init__(self):
        pass

    def add_level(self):
        pass

    def set_mu(self, k, mu):
        pass

    def set_d(self, k, d):
        pass

    def set_classifier(self, k, h_k):
        pass

    def set_threshold(self, k, threshold):
        pass

    def predict(self, X):
        pass

    def error(self, X, y):
        pass

    def gen_error(self, X, y):
        pass


def copy_deep_cascade(obj):
    try:
       return pickle.loads(pickle.dumps(obj, -1))
    except PicklingError:
       return copy.deepcopy(obj)

def get_threshold(clf, mu, kernel, X_train, y_train):
    if mu == 1.0 or not X_train:
        return 0.0
    data = [(X, np.sum(clf.K[:,y[0]] * clf.all_alpha * y[1])) for (X,y) in zip(X_train, enumerate(y_train))]
    data.sort(key=lambda x: x[1])
    thres = 0
    added = 0
    latest = data[0]
    for (x,z) in data:
        latest = (x,z)
        added += 1
        if added / len(X_train) >= mu:
            break
    return latest[1]

class DeepCascades(object):
    def __init__(self, maxL=4, minMu=0.1, maxMu=0.9, muStep=0.25, minD=1, maxD=3, n_passes=5):
        self.maxL = maxL
        self.minMu = minMu
        self.maxMu = maxMu
        self.muStep = muStep
        self.minD = minD
        self.maxD = maxD
        self.n_passes = n_passes
        self.best = None

    def train(self, X_train, y_train):
        X = copy.deepcopy(X_train)
        y = copy.deepcopy(y_train)
        dc = PerceptronDeepCascade()
        self.best = None
        best_error = 1.0
        for k in range(1, self.maxL+1):
            dc.add_level()
            die = False
            for mu in drange(self.minMu, self.maxMu+1e-5, self.muStep):
                if k == self.maxL: # this is the last level, deal with all the remaining samples here
                    mu = 1.0
                    die = True
                dc.set_mu(k, mu)
                for d in range(self.minD, self.maxD+1):
                    dc.set_d(k, d)
                    kernel = lambda x,y: perceptron.polynomial_kernel(x, y, d)
                    h_k = perceptron.KernelPerceptron(kernel=kernel, T=self.n_passes)
                    h_k.fit(np.array(X), np.array(y))
                    dc.set_classifier(k, h_k)
                    dc.set_threshold(k, get_threshold(h_k, mu, kernel, X, y))
                    error = dc.gen_error(X_train, y_train)
                    if error < best_error:
                        self.best = copy_deep_cascade(dc)
                        best_error = error
                if die:
                    break

    def predict(self, X):
        return self.best.predict(X)

    def test(self, X_test, y_test):
        y_predict = self.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))


def load_dataset():
    f = open('breast_cancer.csv', 'rb')
    reader = csv.reader(f)
    rows = [r for r in reader]
    f.close()
    X = []
    Y = []
    for r in rows[1:]:
        x = [float(e) for e in r[:-1]]
        y = int(r[-1])
        if y==0: y=-1
        X.append(x)
        Y.append(y)
    return (X,Y)

if __name__ == "__main__":
    dcs = DeepCascades()
    (X, y) = load_dataset()
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dcs.train(X_train, y_train)
    dcs.test(X_test, y_test)
