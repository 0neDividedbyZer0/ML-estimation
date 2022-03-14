
import numpy as np
import math

# X is an N by D data matrix, y is an N x 1 binary label matrix with 0 or 1.
# the algorithm is O(nlogn). The expected estimate may also be a little bit off by a 10-20
# TODO: convert to full Numpy
def estimateMEC(X, y):
    thresholds = 0
    N, D = X.shape
    table = []
    summed = np.sum(X, axis = 0)
    for i in range(N):
        table.append((summed[i], y[i]))
    sortedTable = table.sort(key = lambda x: x[0])
    c = sortedTable[i][1]
    for i in range(N):
        if sortedTable[i][1] != c:
            c = sortedTable[i][1]
            thresholds += 1
    maxMEC = thresholds * D + thresholds + 1
    expectedMEC = math.log2((thresholds + 1) * D)
    return maxMEC, expectedMEC