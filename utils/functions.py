import numpy as np

def unit_step_func(x):
    return np.where(x>0, 1, 0)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(a):
    return 1 - a**2