import subprocess32
import sys
import random
import os

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print >> sys.stderr, 'Please supply a dataset and gamma. For example: python pdc_multiiprocess ionosphere_train_1:ionosphere_test_1 1e-2'
        sys.exit(1)
    train_file = sys.argv[1]
    test_file = train_file.split(':')[-1]
    train_file = train_file.split(':')[0]
    if train_file == test_file: test_file = None
    gamma = float(sys.argv[2])

    processes = []
    dumps = []

    for L in range(2,4+1):
        dumpname = 'dump_%d_%d' % (L, random.randint(1000,9999))
        dumps.append(dumpname)
        processes.append(subprocess32.Popen(["python", "pdc.py", "-l%d" % L, "-L%d" % L, "-m0.2", "-M0.8", "-s0.2",
            "-g%f" % gamma, "-d1", "-D4", "-f%s" % dumpname, '-t%s' % train_file, '-T%s' % test_file],
            stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr))

    print 'All processes started'

    for p in processes:
        p.wait()

    print 'All processes finished, merging results ...'

    subprocess32.Popen(["python", "compare_dumps.py", "%s:%s" % (train_file, test_file), "-ocascade_%s_%f" % (train_file.split(':')[0], gamma)] + dumps,
            stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr).wait()

    for d in dumps:
        try:
            os.remove(d)
        except OSError:
            pass

    print 'Everything done.'
