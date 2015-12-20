import subprocess32
import sys
import random
import os

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print >> sys.stderr, 'Please supply a dataset prefix and gamma. For example: python crossvalidate.py breast_cancer_train 1e-2'
        sys.exit(1)
    dataset = sys.argv[1]
    gamma = float(sys.argv[2])

    processes = []

    for fold in range(1,3+1):
        train_name = dataset + '_train_%d' % fold
        test_name = dataset + '_test_%d' % fold
        processes.append(subprocess32.Popen(["python", "pdc_multiprocess.py", "%s:%s" % (train_name, test_name), "%f" % gamma],
            stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr))

    print 'All folds started'

    for p in processes:
        p.wait()

    print '3-fold cross-validation done.'

