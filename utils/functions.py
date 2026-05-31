import numpy as np

def unit_step_func(x):
    return np.where(x>0, 1, 0)