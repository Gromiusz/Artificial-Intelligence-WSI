import numpy as np
from gradient_descent import gradient_descent
import utils
import matplotlib.pyplot as plt
import os


# gradient funkcji f(x,y)
def gradient_vector(x):
    return np.array([np.exp(-0.5 * x[0] - x[0]**2 - x[1]**2) * (9 - 4.5 * x[0] - 18 * x[0]**2) * x[1], -9 * np.exp(-0.5 * x[0] - x[0]**2 - x[1]**2) * x[0] * (-1 + 2 * x[1]**2)])

# inicjalizacja stałych
tolerance=1e-08
n_iter = 50

complete_results = np.full((9, 4), None)

# zależność prawidłowych wyników od kroku uczącego
start_points = utils.generate_grid(-2.0, 2.0, 0.1)
for i in range(5):
    prop = 0
    position_results = np.array([None, None])
    value_results = np.array([None, None, None])
    learn_rate=0.025*2**(i)
    position_results = np.array(utils.compute_global_minima_position(gradient_descent, gradient_vector, start_points, learn_rate, n_iter, tolerance))
    value_results = np.array(utils.compute_global_minima_value(position_results))
    prop = utils.count_proper(value_results)
    utils.plot_one(value_results, "Standardowa gęstość siatki, krok uczący: " + str(learn_rate))
    complete_results[i] = [learn_rate, start_points.shape[0], prop, prop/start_points.shape[0]]

# zależność prawidłowych wyników od gęstości siatki
learn_rate=0.2
for i in range(4):
    prop = 0
    start_points = np.array([None, None])
    position_results = np.array([None, None])
    value_results = np.array([None, None, None])
    start_points = utils.generate_grid(-2.0, 2.0, 0.05*2**i)
    position_results = np.array(utils.compute_global_minima_position(gradient_descent, gradient_vector, start_points, learn_rate, n_iter, tolerance))
    value_results = np.array(utils.compute_global_minima_value(position_results))
    prop = utils.count_proper(value_results)
    utils.plot_one(value_results, "Siatka zawierająca " + str(start_points.shape[0]) + " elementów, krok uczący: 0.02")
    complete_results[i+5] = [learn_rate, start_points.shape[0], prop, prop/start_points.shape[0]]

print("[krok_uczacy liczba_el liczba_trafień trafnosc]")
print(complete_results)

learn_rate_results = np.array(complete_results[:5])
grid_results = np.array(complete_results[5:])

# print(learn_rate_results)
# print(grid_results)

utils.plot_variable_results(learn_rate_results, 0, 'Zależność trafności od kroku uczącego', 'wielkość kroku uczącego')
utils.plot_variable_results(grid_results, 1, 'Zależność trafności od gęstości siatki', 'liczba elementów')
