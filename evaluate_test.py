from __future__ import division

import pdc
import cPickle as pickle
import re
from pdc import PerceptronDeepCascade
import glob
import collections
import statistics

pat = re.compile(r'cascade_(.*)_train_([0-9.]+)')

if __name__ == '__main__':
    files = glob.glob('test_results/*')
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
        gamma = float(groups[1])
        #if gamma != 1.00 and gamma != 0.01: continue
        #if origin != 'ionosphere_train': continue
        test_file = origin + '_test'
        train_file = origin + '_train'
        (X_train, y_train, X_test, y_test) = pdc.load_dataset(train_file, test_file)
        print
        print origin, gamma
        print
        (e, mk_m) = dc.error(X_train, y_train)
        (e, mk_m) = dc.error(X_test, y_test)
        results[origin][gamma].append(e)
        if gamma != dc.gamma:
            print 'SHOULD NOT HAPPEN'
            sys.exit(1)

    print '\nCross-validation errors:'
    for (origin, gamma_map) in results.items():
        for (gamma, error) in gamma_map.items():
            print origin, gamma, error
