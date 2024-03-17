import numpy as np
import random
import utils


# Definicja funkcji celu
def objective_function(x, y):
    return 9 * x * y / np.exp(x ** 2 + 0.5 * x + y ** 2)



# Parametry algorytmu
mi = 10  # Rozmiar populacji rodzicielskiej
lambda_ = 50  # Rozmiar populacji potomnej
population_size = mi + lambda_  # Całkowita liczba osobników
num_generations = 50  # Liczba generacji
mutation_sigma = 0.1  # Odchylenie standardowe mutacji
side_length = 2

# Uruchomienie algorytmu ewolucyjnego
# best_individual, best_fitness = utils.evolutionary_strategy(objective_function, mi, lambda_, population_size, side_length, num_generations, mutation_sigma)

best_individual, best_fitness = utils.execute_strategy(objective_function, mi, lambda_, population_size, side_length, num_generations, mutation_sigma)

print("Najlepsze rozwiązanie:", best_individual)
print("Wartość funkcji celu dla najlepszego rozwiązania:", best_fitness)

# print(initialize_population(population_size))
# [np.random.uniform(-5, 5, size=2) for _ in range(population_size)]
# print([np.random.uniform(-3, 3, size=2) for _ in range(2)])
