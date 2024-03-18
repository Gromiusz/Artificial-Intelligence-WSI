from utils import mutate_population
import utils
import numpy as np
import random

local_maxima = np.array([[0.59307, 0.707107, 1.19715],[-0.84307, -0.707107, 2.43687]])
local_minima = np.array([[0.59307, -0.707107, -1.19715],[-0.84307, 0.707107, -2.43687]])

def objective_function(x, y):
    if(abs(x) < 15 and abs(y) < 15):
        return 9 * x * y / np.exp(x ** 2 + 0.5 * x + y ** 2)
    else:
        return 0.09

mu = 10
lambda_ = 60
num_generations = 100
mutation_sigma = 0.1
bounds = [(0, 2, 0, 2), (-2, 0, 0, 2), (-2, 0, -2, 0), (0, 2, -2, 0)]

best_individuals, best_individuals_fitness = utils.execute_strategy(objective_function, mu, lambda_, num_generations, mutation_sigma, bounds, -1)
utils.print_results(best_individuals, best_individuals_fitness, "Minimum", local_minima)
# for i, (best_individual, best_fitness) in enumerate(zip(best_individuals, best_individuals_fitness), 1):
#     print(f"Minimum {i}: {best_individual}   with value: {best_fitness}")

best_individuals, best_individuals_fitness = utils.execute_strategy(objective_function, mu, lambda_, num_generations, mutation_sigma, bounds, 1)
utils.print_results(best_individuals, best_individuals_fitness, "Maksimum", local_maxima)
# for i, (best_individual, best_fitness) in enumerate(zip(best_individuals, best_individuals_fitness), 1):
#     print(f"Maksimum {i}: {best_individual}   with value: {best_fitness}")

# mutation sigma = 0.001 01 1 10
mutation_sigma_list = [0.001, 0.1, 1, 10]
mutated_population_pos=[]
mutated_population_value=[]
mutated_population = []
for sigma in mutation_sigma_list:
    best_individuals, best_individuals_fitness = utils.execute_strategy(objective_function, mu, lambda_, num_generations, sigma, bounds, -1)
    mutated_population_pos.append(best_individuals)
    mutated_population_value.append(best_individuals_fitness)



# mu, lambda = 1,1    1,16    16,1   16,16   128,512
mu_and_lambda = [(1,1), (1,16), (16,1), (16,16), (128,512)]
mu_and_lambda_population = []
mu_and_lambda_population_value = []
for new_mu, new_lambda in mu_and_lambda:
    best_individuals, best_individuals_fitness = utils.execute_strategy(objective_function, new_mu, new_lambda, num_generations, mutation_sigma, bounds, -1)
    mu_and_lambda_population.append(best_individuals)
    mu_and_lambda_population_value.append(best_individuals_fitness)

mu = 128
lambda_ = 512
num_generations = 1000
mutation_sigma = 3
err = np.zeros((2,3))
sum_ = np.zeros(2)

bounds = [(10, 10, 10, 10)]
best_individuals, best_individuals_fitness = utils.execute_strategy(objective_function, mu, lambda_, num_generations, mutation_sigma, bounds, -1)
for i, (best_individual, best_fitness) in enumerate(zip(best_individuals, best_individuals_fitness), 1):
    print(f"Minimum {i}: {best_individual}   with value: {best_fitness}")
    for j in range(2):
        err[j][0] += best_individual[0] - local_minima[j][0]
        err[j][1] += best_individual[1] - local_minima[j][1]
        err[j][2] += best_fitness - local_minima[j][2]
        sum_[j] = sum(abs(err[j][k]) for k in range(3))
    if len(err) > 0:  # Sprawdzenie, czy tablica err nie jest pusta
        choice = int(sum_[0] > sum_[1])
        print(f"Error x = {err[choice][0]}   y = {err[choice][1]}   z = {err[choice][2]}")
    else:
        print("Err is empty.")

    
    




#print(mu_and_lambda_population_value)

# dla mu, lambda = 128,512, sigma=1 ustawic punkt na 10,10