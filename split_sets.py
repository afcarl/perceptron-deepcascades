import csv
import sys
import random
import os

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print >> sys.stderr, 'Please supply a dataset prefix. For example: python split_sets.py breast_cancer'
        sys.exit(1)
    dataset = sys.argv[1]

    f = open(dataset, 'rb')
    reader = csv.reader(f)
    rows = [r for r in reader]
    random.shuffle(rows)
    n = len(X_train)
    split = int(n * 0.8)

    for r in rows:
        for i in range(len(r)):
            if r[i]=='?': r[i] = 0

    f = open(dataset + '_train', 'wb')
    csv.writer(f).writerows(rows[:split])
    f.close()

    f = open(dataset + '_test', 'wb')
    csv.writer(f).writerows(rows[split:])
    f.close()

