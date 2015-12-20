import subprocess32
import sys
import random
import os

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print >> sys.stderr, 'Please supply a gamma.'
        sys.exit(1)
    gamma = float(sys.argv[1])

    processes = []
    dumps = []

    for L in range(2,4+1):
        dumpname = 'dump_%d_%d' % (L, random.randint(1000,9999))
        dumps.append(dumpname)
        processes.append(subprocess32.Popen(["python", "pdc.py", "-l%d" % L, "-L%d" % L, "-m0.2", "-M0.8", "-s0.2",
            "-g%.1f" % gamma, "-d1", "-D4", "-f%s" % dumpname], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr))

    print 'All processes started'

    for p in processes:
        p.wait()

    print 'All processes finished, merging results ...'

    subprocess32.Popen(["python", "compare_dumps.py", "-ocascade"] + dumps, stdin=sys.stdin, stdout=sys.stdout,
            stderr=sys.stderr).wait()

    for d in dumps:
        try:
            os.remove(d)
        except OSError:
            pass

    print 'Everything done.'
