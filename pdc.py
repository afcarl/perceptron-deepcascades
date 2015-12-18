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
        self.mu = {}
        self.d = {}
        self.h = {}
        self.threshold = {}

    def set_mu(self, k, mu):
        self.mu[k] = mu

    def set_d(self, k, d):
        self.d[k] = d

    def set_classifier(self, k, h_k):
        self.h[k] = h_k

    def set_threshold(self, k, threshold):
        self.threshold[k] = threshold

    def cascade_info(self):
        print 'mu: ' + str(self.mu)
        print 'd: ' + str(self.d)
        print 'threshold: ' + str(self.threshold)

    def predict(self, X):
        y = []
        for x in X:
            k = 1
            result = None
            while result is None:
                x_pred = np.array(x)
                project = self.h[k].project(np.array([x_pred]))
                dist = abs(project)
                if dist >= self.threshold[k]:
                    result = np.sign(project)
                    break
                k += 1
            y.append(int(result[0]))
        return np.array(y)

    def error(self, X, y):
        y_predict = self.predict(X)
        incorrect = np.sum(y_predict != np.array(y))
        e = incorrect / len(y_predict)
        self.cascade_info()
        print 'Error: ' + str(e)
        return e

    def gen_error(self, X, y):
        return self.error(X, y) # TODO Implement generalization error


def get_threshold(clf, mu, kernel, X_train, y_train):
    if mu >= 1.0 or not X_train:
        return 0.0
    projections = clf.project(X_train)
    data = [(X, y, p) for (X,y,p) in zip(X_train, y_train, projections)]
    data.sort(key=lambda x: x[2])
    thres = 0
    added = 0
    latest = None
    for (x,y,p) in data:
        latest = (x,y,p)
        added += 1
        if added / len(X_train) >= mu:
            break
    t = abs(latest[2])
    X = [d[0] for d in data[:added+1]]
    y = [d[1] for d in data[:added+1]]
    return (X,y,t)

class DeepCascades(object):
    def __init__(self, minL=2, maxL=4, minMu=0.25, maxMu=0.75, muStep=0.25, minD=1, maxD=3, n_passes=4):
        self.minL = minL
        self.maxL = maxL
        self.minMu = minMu
        self.maxMu = maxMu
        self.muStep = muStep
        self.minD = minD
        self.maxD = maxD
        self.n_passes = n_passes
        self.best = None

    def train_cascade(self, X_train, y_train, L, mu, d):
        X = copy.deepcopy(X_train)
        y = copy.deepcopy(y_train)
        dc = PerceptronDeepCascade()
        for k in range(1, L+1):
            dc.set_mu(k, mu[k-1])
            dc.set_d(k, d[k-1])
            kernel = lambda x,y: perceptron.polynomial_kernel(x, y, d[k-1])
            h_k = perceptron.KernelPerceptron(kernel=kernel, T=self.n_passes)
            h_k.fit(np.array(X), np.array(y))
            dc.set_classifier(k, h_k)
            if k==L:
                dc.set_threshold(k, 0.0)
                return dc
            else:
                X,y,t = get_threshold(h_k, mu[k-1], kernel, X, y)
                dc.set_threshold(k, t)

    def mu_permutations(self, L):
        if L==0:
            yield []
        else:
            for p in self.mu_permutations(L-1):
                for mu in drange(self.minMu, self.maxMu+1e-5, self.muStep):
                    yield [mu] + p

    def d_permutations(self, L):
        if L==0:
            yield []
        else:
            for p in self.d_permutations(L-1):
                for d in range(self.minD, self.maxD+1):
                    yield [d] + p

    def train(self, X_train, y_train):
        best_error = 1.0
        for L in range(self.minL, self.maxL+1):
            for mu in self.mu_permutations(L):
                for d in self.d_permutations(L):
                    dc = self.train_cascade(X_train, y_train, L, mu, d)
                    error = dc.gen_error(X_train, y_train)
                    if error < best_error:
                        self.best = dc
                        best_error = error

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
