import os
import random

import torch
import numpy as np

def range_to_world(ranges, bin_length):
    '''
    ## Convert range bin indices into world coords
    ### (radial distances in meters)
    将射线上的距离单位映射到真实世界
    '''
    return ranges * bin_length - (bin_length/2.0)