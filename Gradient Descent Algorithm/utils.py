import numpy as np
import matplotlib.pyplot as plt
import os


def generate_grid(start, end, gap):
# tworzenie punktów startowych w postaci siatki równo oddalonych punktów
    x = np.arange(start, end, gap)
    y = np.arange(start, end, gap)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.flatten(), yy.flatten()))
    return grid_points

def compute_global_minima_position(function, gradient, start_array, learn_rate, n_iter, tolerance):
# na podstawie wejścia (siatki starowej) wyznaczany jest zbiór punktów wyjściowych. Punkty startowe jako "kulki wpadają do dołków (minimów lokalnych)"
    k = start_array.shape[0]
    result = [None] * start_array.shape[0]
    for i in range(start_array.shape[0]):
        result[i] = function(gradient, start_array[i], learn_rate, n_iter, tolerance)
    return result

def compute_global_minima_value(position_tab):
# wyznacza i dodaje kolumnę z wartością funkcji dla danej pozycji (x,y)
    pos_and_value = position_tab
    pos_and_value = [[*row, None] for row in position_tab]
    for i in range(position_tab.shape[0]):
        x = position_tab[i][0]
        y = position_tab[i][1]
        pos_and_value[i][2] = 9 * x * y / np.exp(x**2 + 0.5*x + y**2)
    return pos_and_value

def count_proper(results):
# zlicza trafienia punktow po wykonaniu algorytmow w punkt zgodny z obliczeniami analitycznymi
    proper=0
    for i in range(results.shape[0]):
        if results[i,0] <= 0.59308 and results[i,0] >= 0.59306:
            if results[i,1] <= -0.70709 and results[i,1] >= -0.70711:
                if results[i,2] <= -1.19714 and results[i,2] >= -1.19716:
                    proper=proper+1
        elif results[i,0] <= -0.84304 and results[i,0] >= -0.84306:
            if results[i,1] <= 0.70711 and results[i,1] >= 0.70709:
                if results[i,2] <= -2.43686 and results[i,2] >= -2.43688:
                    proper=proper+1
    return proper
        
def plot_one(results, title: str):
    plt.clf()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(results[:,0], results[:,1], results[:,2])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title(title)
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/')
    file_name = f"{title.replace(' ', '_')}.png"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, file_name))


def plot_variable_results(complete_results, num_of_var_column, title: str, var_str: str):
    plt.clf()
    plt.plot(complete_results[:,num_of_var_column], complete_results[:,3]*100, marker='o')
    plt.xlabel(var_str)
    plt.ylabel('trafność [%]')
    plt.title(title)
    plt.ylim(0, 50)
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/')
    file_name = f"{title.replace(' ', '_')}.png"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, file_name))