import subprocess32
import sys
import random

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print >> sys.stderr, 'Please supply a gamma.'
        sys.exit(1)
    gamma = float(sys.argv[1])

    processes = []
    dumps = []

    for mu in [0.2, 0.4, 0.6, 0.8]:
        dumpname = 'dump_%.1f_%d' % (mu, random.randint(1000,9999))
        dumps.append(dumpname)
        processes.append(subprocess32.Popen(["python", "pdc.py", "-l2", "-L4", "-m%.1f" % mu, "-M%.1f" % mu, "-g%.1f" % gamma, "-d1", "-D4",
            "-f%s" % dumpname], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr))

    print 'All processes started'

    for p in processes:
        p.wait()

    print 'All processes finished, merging results ...'

    subprocess32.Popen(["python", "compare_dumps.py", "-ocascade"] + dumps, stdin=sys.stdin, stdout=sys.stdout,
            stderr=sys.stderr)

    print 'Everything done.'
