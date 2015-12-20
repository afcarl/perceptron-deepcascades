from __future__ import division

import pdc
import cPickle as pickle
import re
from pdc import PerceptronDeepCascade
import glob
import collections

pat = re.compile(r'cascade_(.*)_train_([0-9])_([0-9.]+)')

if __name__ == '__main__':
    files = glob.glob('results/*')
    results = collections.defaultdict(lambda: collections.defaultdict(list))
    for fname in files:
        dc = None
        try:
            f = open(fname, 'rb')
            dc = pdc.load_cascade(f)
            f.close()
        except IOError:
            print >> sys.stderr, 'Dump file "%s" does not exist.' % fname
            sys.exit(1)
        except pickle.PickleError:
            print >> sys.stderr, 'Dump file "%s" does not have a valid cascade dump.' % fname
            sys.exit(1)
        fname = fname.split('/')[-1]
        groups = pat.match(fname).groups()
        origin = groups[0]
        fold = int(groups[1])
        gamma = float(groups[2])
        test_file = origin + '_test_%d' % fold

        (X_train, y_train, X_test, y_test) = pdc.load_dataset(test_file, test_file)
        (e, mk_m) = dc.error(X_test, y_test)
        results[origin][gamma].append(e)

    print '\nCross-validation errors:'
    for (origin, gamma_map) in results.items():
        for (gamma, error) in gamma_map.items():
            print origin, gamma, sum(error)/len(error)
