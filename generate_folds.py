import csv
import random
import sys

def write_csv(fname, n, train, test):
    f = open(fname+'_train_%d' % n, 'wb')
    writer = csv.writer(f)
    writer.writerows(train)
    f.close()
    f = open(fname+'_test_%d' % n, 'wb')
    writer = csv.writer(f)
    writer.writerows(test)
    f.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print >> sys.stderr, 'Please enter the dataset filename.'
        sys.exit(1)
    fname = sys.argv[1]
    f = open(fname, 'rb')
    reader = csv.reader(f)
    rows = [r for r in reader]
    f.close()
    random.shuffle(rows)
    n = len(rows)
    split = int(n/3)
    split1 = rows[:split]
    split2 = rows[split:]
    split3 = split2[split:]
    split2 = split2[:split]

    train1 = split2 + split3
    test1 = split1

    train2 = split1 + split3
    test2 = split2

    train3 = split1 + split2
    test3 = split3

    prefix = fname.split('.')[0]
    write_csv(prefix, 1, train1, test1)
    write_csv(prefix, 2, train2, test2)
    write_csv(prefix, 3, train3, test3)

