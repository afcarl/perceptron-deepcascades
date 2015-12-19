import perceptron
import csv
import numpy as np
import pylab as pl

class PerceptronTest(object):
    def __init__(self, passes=5000):
        self.pclf = perceptron.KernelPerceptron(kernel=perceptron.linear_kernel, T=passes)

    def train(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.pclf.fit(self.X_train, self.y_train)

    def plot_margin(self, X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")

        # w.x + b = 0
        a0 = -4; a1 = f(a0, clf.w, clf.b)
        b0 = 4; b1 = f(b0, clf.w, clf.b)
        pl.plot([a0,b0], [a1,b1], "k")

        pl.axis("tight")
        pl.show()

    def test(self, X_test, y_test):
        y_predict = self.pclf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))
        self.plot_margin(self.X_train[self.y_train==1], self.X_train[self.y_train==-1], self.pclf)
        self.plot_margin(np.array(X_test)[np.array(y_test)==1], np.array(X_test)[np.array(y_test)==-1], self.pclf)


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
        X.append(x)
        Y.append(y)
    return (X,Y)

if __name__ == "__main__":
    dc = PerceptronTest()
    (X, y) = load_dataset()
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    def gen_lin_separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_non_lin_separable_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([5, 5])
        mean2 = np.array([9, 9])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    X1, y1, X2, y2 = gen_lin_separable_overlap_data()

#    dc.train(X_train, y_train)
#    dc.test(X_test, y_test)

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)

    dc.train(X_train, y_train)
    dc.test(X_test, y_test)
