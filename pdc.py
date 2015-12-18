import perceptron
import csv
import numpy as np

class DeepCascades(object):
    def __init__(self):
        pass

    def train(self, X_train, y_train):
        pass

    def test(self, X_test, y_test):
        pass


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
    dc = DeepCascades()
    (X, y) = load_dataset()
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dc.train(X_train, y_train)
    dc.test(X_test, y_test)
