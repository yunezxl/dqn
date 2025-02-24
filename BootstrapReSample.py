import random
import numpy as np

def expand_demand(SmallData, ExpandSize):
    dim0 = len(SmallData)
    dim1 = len(SmallData[0])
    resampled_data = [[] for _ in range(dim0)]
    SeqLocVec = np.arange(0, dim1)
    while len(resampled_data[0]) < ExpandSize:
        temp_loc = random.choice(SeqLocVec)
        for i in range(dim0):
            resampled_data[i].append(SmallData[i][temp_loc])

    return resampled_data