from __future__ import division

import numpy as np
from sklearn import cross_validation, svm
import pdc
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print >> sys.stderr, 'Please specify a training and testing file'
        sys.exit(1)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    (X_train, y_train, X_test, y_test) = pdc.load_dataset(train_file, test_file)

    all_scores = []
    C_s = np.array([0.001, 0.1, 1.0, 10.0, 100.0])
    for d in range(1,4+1):
        svc = svm.SVC(kernel='poly', degree=d)
        scores = list()
        scores_std = list()
        for C in C_s:
            svc.C = C
            this_scores = cross_validation.cross_val_score(svc, X_train, y_train, n_jobs=1, cv=3)
            scores.append(np.mean(this_scores))
            scores_std.append(np.std(this_scores))
            all_scores.append((1.0-np.mean(this_scores), d, C))
    all_scores.sort(key=lambda x: x[0])

    best_d = all_scores[0][1]
    print 'best d:', best_d
    svc = svm.SVC(kernel='poly', degree=best_d)
    all_scores = []
    for C in C_s:
        svc.C = C
        this_scores = 1.0-cross_validation.cross_val_score(svc, X_train, y_train, n_jobs=1, cv=3)
        all_scores.append((np.mean(this_scores), np.std(this_scores), C, best_d))

    all_scores.sort(key=lambda x: x[0])
    print 'Best result: %f +/- %f' % (all_scores[0][0], all_scores[0][1])
    d = best_d
    C = all_scores[0][2]
    print 'd=',d,'C=',C

    print 'Test error'
    clf = svm.SVC(kernel='poly', degree=d, C=C)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print "%d out of %d predictions correct" % (correct, len(y_predict))
    print 'Error: %f' % (1.0 - (correct/len(y_predict)))
