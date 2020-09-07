# Use: python improv.py file.out

from __future__ import division, print_function

import sys
import re
import numpy as np
import glob
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

def get_improv_list(fname, block):
    # fname = sys.argv[1]
    # print(fname)
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()

    # # Paramaters
    # block = 0

    init = [(l, i) for i, l in enumerate(lines) if l.startswith('c Deci Hard:')]
    # print(init)
    assert len(init) == 1
    init = init[0]
    # print(init)

    regex = r"^c Deci Hard: \d+ Soft: (\d+)"
    match = re.search(regex, init[0])
    deci_soft = int(match.group(1))
    # print(deci_soft)

    # final = [(l, i) for i, l in enumerate(lines) if l.startswith('c Original Hard:')]
    # assert len(final) == 1
    # final = final[0]

    # best = [(l, i) for i, l in enumerate(lines) if l.startswith('c Best Block:')]
    # assert len(best) == 1
    # best = best[0]

    updates = [l for l in lines[init[1]:] if l.startswith('c Block %d: Updated best model' % block)]
    # print(updates)

    regex = r"^c Block (\d+): Updated best model\((\d+), (\d+)\) at (\d+)\n$"
    matches = [re.search(regex, l) for l in updates]
    # print(matches)
    matches = [(int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))) for match in matches]
    # print(matches)

    if len(matches) == 0:
        # result = np.array([[i,0] for i in range(101)]).astype(np.float)
        # return result
        return None
            # print("Exiting: ",fname)
            # exit(0)

    total_iters = matches[-1][3] + 1


    # print(0, 0, sep=',')
    # for a in matches:
            # print((a[3] + 1) * 100 / total_iters, (a[2] - deci_soft) / deci_soft * 100, sep=',')
    # print(total_iters)

    result = []
    result.append([0,0])
    for a in matches:
            result.append([(a[3] + 1) * 100 / total_iters, (a[2] - deci_soft) / deci_soft * 100])
    return np.array(result)
    # print(result)
    # print(len(result))

def filter_to_100(inp):
    result = []
    input_idx = 0
    for percentage in range(0,101):
        while input_idx < len(inp) and inp[input_idx][0] <= percentage:
            input_idx += 1
        result.append(inp[input_idx-1][1])
    return np.array(result)

def get_max_over_blocks(fname, blocks):
    max_over_blocks = np.zeros(101)
    count_zero = 0
    for block in blocks:
        lis = get_improv_list(fname, block)
        if lis is None:
            count_zero += 1
            continue
        res = filter_to_100(lis)
        max_over_blocks = np.amax(np.vstack((res, max_over_blocks)),axis=0)
    if count_zero == len(blocks):
        no_improv = True
    else:
        no_improv = False
    return max_over_blocks, no_improv

def get_avg_over_benchmarks(fnames,blocks):
    num_benchmarks = len(fnames)
    assert num_benchmarks >= 1
    result = np.zeros(101)
    count_no_improv = 0
    for fname in fnames:
        max_over_blocks, no_improv = get_max_over_blocks(fname, blocks)
        # print(max_over_blocks)
        if no_improv:
            count_no_improv += 1
        else:
            result += max_over_blocks
    result /= num_benchmarks
    return result, count_no_improv


if __name__ == "__main__":
    dirpath = sys.argv[1]
    blocks = range(int(sys.argv[2]))
    outpath = sys.argv[3]
    benchmark_types = ["benchmarks_", "benchmarks-w1", "benchmarks-w2", "gauss-100_", "gauss-10000_"]
    
    for btype in benchmark_types:
        paths = glob.glob(dirpath + "/" + btype + "*.out")
        # print(paths)
        result, count_no_improv = get_avg_over_benchmarks(paths, blocks)
        # print(outpath)
        # print(outpath + "/" + btype + "_log.out")
        with open(outpath + "/" + btype + "_log.out", "w") as f:
            f.write(str(list(result)) + '\n')
            f.write(str(count_no_improv))
    


# fname = sys.argv[1]
# f = open(fname, 'r')
# lines = f.readlines()
# f.close()

# # Paramaters
# blocks = [0,1,2,3]

# max_over_blocks = np.zeros(101)

# for block in blocks:
    # lis = get_improv_list(fname, block)
    # print("List:",lis)
    # res = filter_to_100(lis)
    # print("Result:", res)
    # print("VStack:", np.vstack((res,max_over_blocks)))
    # print("HStack:", np.hstack((res, max_over_blocks)))
    # print(res.shape)
    # print(max_over_blocks.shape)
    # print(np.vstack((res, max_over_blocks)).shape)
    # max_over_blocks = np.amax(np.vstack((res, max_over_blocks)),axis=0)
    # print("MOB:",max_over_blocks)

# print(max_over_blocks)
# print(max_over_blocks.shape)
# print(res.shape)
