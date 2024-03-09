import numpy as np

def gradient_descent(
    gradient, start, learn_rate, n_iter=50, tolerance=1e-06
):
    position = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(position)
        if np.all(np.abs(diff) <= tolerance):
            break
        position += diff
    return position