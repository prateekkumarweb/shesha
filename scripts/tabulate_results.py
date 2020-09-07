#!/bin/python

import sys

with open(sys.argv[1]) as f:
    outs = []
    for line in f:
        if len(line) > 0 and line[0] == 'o':
            outs.append(line)
    if len(outs) == 0:
        # print(",")
        print("")
    else:
        o1 = outs[0].split()[1]
        o2 = outs[-1].split()[1]
        o1 = int(o1)
        o2 = int(o2)
        # print("{},{}".format(o1,o2))
        print(o2)