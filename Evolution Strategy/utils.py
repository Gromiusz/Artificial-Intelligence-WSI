import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os

def initialize_population(population_size, bounds):
    return np.random.uniform(low=(bounds[0], bounds[2]), high=(bounds[1], bounds[3]), size=(population_size, 2))

def mutate_population(population, sigma):
    noiseMatrix = np.random.normal(0, sigma, (len(population), 2))
    return population + noiseMatrix

def interpolation_crossover(parent1, parent2):
    a = np.random.uniform(0, 1)
    return a * parent1 + (1 - a) * parent2

def calculate_function_value(objective_function, population):
    return [objective_function(*individual) for individual in population]

def select_best_individuals(population, population_values, num_best):
# strategia selekcji to wybranie osobników z najmniejszą wartością funkcji
    sorted_indices = np.argsort(population_values)
    return [population[i] for i in sorted_indices[:num_best]]

def select_not_only_best_individuals(population, population_values, num_best):
# strategia selekcji to wybranie osobników równiez gorszych jakościowo
    sorted_indices = np.argsort(population_values)
    half_num_lowest = math.ceil(num_best/10)
    best_indices = sorted_indices[:half_num_lowest]
    rest_indices = np.setdiff1d(sorted_indices, best_indices)
    random_indices = np.random.choice(rest_indices, size=num_best-half_num_lowest, replace=False)
    best_indices = np.append(best_indices, random_indices)
    return [population[i] for i in best_indices[:num_best]]

def select_best_individuals_descending(population, population_values, num_best):
# wybór osobników z największą wartością funkcji
    sorted_indices = np.argsort(population_values)[::-1]
    return [population[i] for i in sorted_indices[:num_best]]

def evolutionary_strategy(objective_function, mi, lambda_, num_generations, mutation_sigma, bounds, flags):
    population_size = mi
    population = initialize_population(population_size, bounds)
    
    for ii in range(num_generations):
        if ii>900:
            mutation_sigma=0.1
            if ii>980:
                mutation_sigma=0.05
        
        #mutacja
        new_generation = mutate_population(population, mutation_sigma)
        new_generation_value = calculate_function_value(objective_function, new_generation)

        for _ in range(lambda_):
            #krzyowanie
            parent1, parent2 = random.choices(new_generation, k=2)
            child = interpolation_crossover(parent1, parent2)
            new_generation = np.vstack([new_generation, child])
            new_generation_value.append(objective_function(*child))
        
        # wybór najmniejszych osobników lub największych w zalezności od wartości flagi
        if flags==-2:
            population = select_not_only_best_individuals(new_generation, new_generation_value, mi)
        elif flags==1:
            population = select_best_individuals_descending(new_generation, new_generation_value, mi)
        elif flags==-1:
            population = select_best_individuals(new_generation, new_generation_value, mi)
    
    #wybór jednego najlepszego osobnika końcowego
    if flags==-1 or flags==-2:
        best_individual = min(population, key=lambda x: objective_function(*x))
    if flags==1:
        best_individual = max(population, key=lambda x: objective_function(*x))

    best_value = objective_function(*best_individual)
    return best_individual, best_value

def execute_strategy(objective_function, mi, lambda_, num_generations, mutation_sigma, bounds, flags):
    #wykonanie alegorytmu dla rónych obszarów startowych. Tutaj są to ćwiartki układu współrzędnych
    best_individuals = []
    best_individuals_value = []
    
    for bound in bounds:
        best_individual, best_value = evolutionary_strategy(objective_function, mi, lambda_, num_generations, mutation_sigma, bound, flags)
        best_individuals.append(best_individual)
        best_individuals_value.append(best_value)
    
    best_individuals_filtered = [ind for ind, val in zip(best_individuals, best_individuals_value) if abs(val) > 1e-1]
    best_individuals_value_filtered = [val for val in best_individuals_value if abs(val) > 1e-1]
    
    return best_individuals_filtered, best_individuals_value_filtered
    # return best_individuals, best_individuals_value

def print_results(best_individuals, best_individuals_fitness, what: str, local_what):
    errors = []
    for i, (best_individual, best_fitness) in enumerate(zip(best_individuals, best_individuals_fitness), 1):
        sum_ = np.zeros(2)
        err = np.zeros((2,3))
        print(f"{what} {i}: {best_individual}   with value: {best_fitness}")
        for j in range(2):
            err[j][0] += best_individual[0] - local_what[j][0]
            err[j][1] += best_individual[1] - local_what[j][1]
            err[j][2] += best_fitness - local_what[j][2]
            sum_[j] = sum(abs(err[j][k]) for k in range(3))
        if len(err) > 0:
            choice = int(sum_[0] > sum_[1])
            print(f"Error x = {err[choice][0]}   y = {err[choice][1]}   z = {err[choice][2]}")
            errors.append(err[choice])
        else:
            print("Err array is empty.")
    return errors
