import pdc
import sys
import cPickle as pickle
from pdc import PerceptronDeepCascade

if __name__ == '__main__':
    args = sys.argv[1:]
    dump = None
    dcs = []

    train_file = args[0]
    test_file = train_file.split(':')[-1]
    train_file = train_file.split(':')[0]
    if train_file == test_file: test_file = None

    for fname in args[1:]:
        try:
            if fname.startswith('-o'):
                dump = fname[2:]
            else:
                f = open(fname, 'rb')
                dcs.append(pdc.load_cascade(f))
                f.close()
        except IOError:
            print >> sys.stderr, 'Dump file "%s" does not exist.' % fname
            sys.exit(1)
        except pickle.PickleError:
            print >> sys.stderr, 'Dump file "%s" does not have a valid cascade dump.' % fname
            sys.exit(1)

    if not dcs:
        print >> sys.stderr, 'Syntax: python compare_dumps.py train_file[:test_file] -ooutdump indump1 indump2 ...'
        sys.exit(1)

    (X_train, y_train, X_test, y_test) = pdc.load_dataset(train_file, test_file)

    dcs = pdc.DeepCascades(cascades=dcs, dump=dump)
    dcs.find_best(X_train, y_train)
    dcs.test(X_test, y_test)

