from __future__ import division

import perceptron
import csv
import numpy as np
from sklearn import preprocessing
import collections
from scipy import misc
import math
import cPickle as pickle
import sys
import getopt

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def vc_dimension(n_features, degree):
    return misc.comb(n_features + degree, degree, exact=True)


class PerceptronDeepCascade(object):
    def __init__(self, gamma):
        self.gamma = gamma
        self.mu = {}
        self.d = {}
        self.h = {}
        self.mk = collections.defaultdict(int)
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

    def predict(self, X, y_true=None):
        y = []
        totals = collections.defaultdict(int)
        (self.T, self.n_features) = X.shape
        for i,x in enumerate(X):
            k = 1
            result = None
            while result is None:
                project = self.h[k].project(np.array([x]))
                dist = abs(project)
                if dist >= self.threshold[k]:
                    result = np.sign(project)
                k += 1
            y_pred = int(result[0])
            y.append(y_pred)
            if y_true is None or y_true[i] == y_pred:
                totals[k-1] += 1
            self.mk[k-1] += 1
        mk_m = []
        k = 1
        while k in totals:
            mk_m.append(totals[k] / len(X))
            k += 1
        print (y_true is not None and 'm_k+:   ' or 'm_k:   ') + str(totals)
        print (y_true is not None and 'm_k+/m: ' or 'm_k/m: ') + str(mk_m)
        return (np.array(y), np.array(mk_m))

    def error(self, X, y):
        (y_predict, mk_m) = self.predict(X, y)
        incorrect = np.sum(y_predict != y)
        e = incorrect / len(y_predict)
        self.cascade_info()
        print 'Training error: ' + str(e)
        if e >= 0.5:
            print 'TRAINING ERROR OVER 50%!'
        return (e, mk_m)

    def vc_dim_bound(self, k):
        dim = vc_dimension(self.n_features, self.d[k])
        if self.mk[k] < dim:
            return math.sqrt(2 * math.log(2))
        else:
            return math.sqrt((2 * dim * math.log(math.e * self.mk[k] / dim))/self.mk[k])

    def gen_error(self, X, y):
        (r, mk_m) = self.error(X, y)
        s = 0.0
        L = len(mk_m)
        for (i, mkm) in enumerate(mk_m):
            k = i+1
            lsum = self.vc_dim_bound(k)
            d_k = k<L and k or k-1
            for j in range(1, d_k+1):
                lsum += self.vc_dim_bound(j)
            s += min(4.0 * self.gamma * lsum, mkm)
        gerr = r + s
        print 'Generalization error: ' + str(gerr)
        return gerr


def get_threshold(clf, mu, kernel, X_train, y_train):
    if mu >= 1.0 or not len(X_train):
        return 0.0
    projections = clf.project(X_train)
    data = [(X, y, abs(p)) for (X,y,p) in zip(X_train, y_train, projections)]
    data.sort(key=lambda x: x[2])
    thres = 0
    added = 0
    latest = None
    for (x,y,p) in data:
        latest = (x,y,p)
        added += 1
        if added / len(X_train) >= mu:
            break
    t = latest[2]
    X = np.array([d[0] for d in data[:added-1]])
    y = np.array([d[1] for d in data[:added-1]])
    return (X,y,t)

def dump_cascade(dc, f):
    for h in dc.h.values():
        h.kernel = None
    pickle.dump(dc, f, -1)
    for (k,h) in dc.h.items():
        h.kernel = perceptron.poly_kernel(dc.d[k])

def load_cascade(f):
    dc = pickle.load(f)
    for (k,h) in dc.h.items():
        h.kernel = perceptron.poly_kernel(dc.d[k])
    return dc


class DeepCascades(object):
    def __init__(self, minL=None, maxL=None, minMu=None, maxMu=None, muStep=None, minD=None, maxD=None, n_passes=None,
                       increasing_d=None, gamma=None, dump=None, cascades=None):
        self.minL = minL
        self.maxL = maxL
        self.minMu = minMu
        self.maxMu = maxMu
        self.muStep = muStep
        self.minD = minD
        self.maxD = maxD
        self.n_passes = n_passes
        self.increasing_d = increasing_d
        self.gamma = gamma
        self.dump = dump
        self.best = None
        self.cascades = cascades

    def train_cascade(self, X_train, y_train, L, mu, d):
        X = np.array(X_train)
        y = np.array(y_train)
        n_samples, n_features = X.shape
        dc = PerceptronDeepCascade(self.gamma)
        for k in range(1, L+1):
            dc.set_d(k, d[k-1])
            kernel = perceptron.poly_kernel(d[k-1])
            h_k = perceptron.KernelPerceptron(kernel=kernel, T=self.n_passes)
            h_k.fit(X, y)
            dc.set_classifier(k, h_k)
            if k==L:
                dc.set_threshold(k, 0.0)
                return dc
            else:
                dc.set_mu(k, mu[k-1])
                X,y,t = get_threshold(h_k, mu[k-1], kernel, X, y)
                if not len(y):
                    X = np.zeros((1, n_features))
                    y = np.array([1])
                dc.set_threshold(k, t)

    def mu_permutations(self, L):
        if L==0:
            yield []
        else:
            for p in self.mu_permutations(L-1):
                for mu in drange(self.minMu, self.maxMu+1e-5, self.muStep):
                    yield p + [mu]

    def d_permutations(self, L):
        if L==0:
            yield []
        else:
            for p in self.d_permutations(L-1):
                start = self.minD
                if self.increasing_d and p:
                    start = p[-1]
                for d in range(start, self.maxD+1):
                    yield p + [d]

    def train(self, X_train, y_train):
        times = 0
        best_error = 2.0
        for L in range(self.minL, self.maxL+1):
            for mu in self.mu_permutations(L-1):
                for d in self.d_permutations(L):
                    dc = self.train_cascade(X_train, y_train, L, mu, d)
                    times += 1
                    error = dc.gen_error(X_train, y_train)
                    if error < best_error:
                        self.best = dc
                        best_error = error
        print 'Generated %d cascades' % (times,)
        if self.dump is not None:
            print 'Writing best model to dump file'
            f = open(self.dump, 'wb')
            dump_cascade(self.best, f)
            f.close()
            print 'Dumped.'

    def find_best(self, X_train, y_train):
        print 'Finding the best cascade from %d cascades ...' % len(self.cascades)
        best_error = 2.0
        for dc in self.cascades:
            error = dc.gen_error(X_train, y_train)
            if error < best_error:
                self.best = dc
                best_error = error
        if self.dump is not None:
            print 'Writing best model to dump file'
            f = open(self.dump, 'wb')
            dump_cascade(self.best, f)
            f.close()
            print 'Dumped.'

    def predict(self, X):
        (y, mk) = self.best.predict(X)
        return y

    def test(self, X_test, y_test):
        print 'Best cascade:'
        self.best.cascade_info()
        y_predict = self.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))


class PerceptronDummyCascade(object):
    def __init__(self, d=3, n_passes=4):
        self.d = d
        self.n_passes = n_passes
        self.clf = None

    def train(self, X_train, y_train):
        self.clf = perceptron.KernelPerceptron(kernel=perceptron.poly_kernel(self.d), T=self.n_passes)
        self.clf.fit(X_train, y_train)

    def predict(self, X):
        return self.clf.predict(X)

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

def split_dataset(X,y):
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = np.array(X[:split]), np.array(X[split:])
    y_train, y_test = np.array(y[:split]), np.array(y[split:])
    print 'Scaling the dataset ...'
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print 'Scaled.'
    return (X_train, y_train, X_test, y_test)

def usage(out=sys.stderr):
    print >> out, '''Example 1: python pdc.py -l2 -L4 -m0.1 -M1.0 -g10e-3 -fdump.dat
Example 2: python pdc.py --minL=2 --maxL=4 --minMu=0.1 --maxMu=1.0 --gamma=10e-3 --dump-file=dump.dat

See pdc.py for more options.
'''


if __name__ == "__main__":
    # defaults
    minL = 2
    maxL = 4
    minMu = 0.2
    maxMu = 0.8
    muStep = 0.2
    minD = 1
    maxD = 4
    n_passes = 4
    increasing_d = True
    gamma = 1e-3
    dump = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], "l:L:m:M:s:d:D:p:ng:f:h", ["minL=", "maxL=", "minMu=", "maxMu=",
                                                 "muStep=", "minD=", "maxD=", "passes=", "not-increasing-h", "gamma=",
                                                 "dump-file=", "--help"])
        for opt, arg in opts:
            if opt in ("-l", "--minL"):
                minL = int(arg)
            if opt in ("-L", "--maxL"):
                maxL = int(arg)
            if opt in ("-m", "--minMu"):
                minMu = float(arg)
            if opt in ("-M", "--maxMu"):
                maxMu = float(arg)
            if opt in ("-s", "--muStep"):
                muStep = float(arg)
            if opt in ("-d", "--minD"):
                minD = int(arg)
            if opt in ("-D", "--maxD"):
                maxD = int(arg)
            if opt in ("-p", "--passes"):
                n_passes = int(arg)
            if opt in ("-n", "--not-increasing-h"):
                increasing_d = False
            if opt in ("-g", "--gamma"):
                gamma = float(arg)
            if opt in ("-f", "--dump-file"):
                dump = arg
            if opt in ("-h", "--help"):
                usage(out=sys.stdout)
                sys.exit()
        if not opts:
            usage(out=sys.stdout)
    except (getopt.GetoptError):
        usage()
        sys.exit(2)

    (X, y) = load_dataset()
    (X_train, y_train, X_test, y_test) = split_dataset(X, y)

    dcs = DeepCascades(minL=minL, maxL=maxL, minMu=minMu, maxMu=maxMu, muStep=muStep, minD=minD, maxD=maxD,
                       n_passes=n_passes, increasing_d=increasing_d, gamma=gamma, dump=dump)
    dcs.train(X_train, y_train)
    dcs.test(X_test, y_test)

