import numpy as np
import random

# Inicjalizacja populacji
def initialize_population(population_size, side_length):
    return [np.random.uniform(-side_length, side_length, size=2) for _ in range(population_size)]

def initialize_population2(population_size, min_x, min_y, max_x, max_y):
    return np.random.uniform(low=(min_x, min_y), high=(max_x, max_y), size=(population_size, 2))

# Mutacja osobnika za pomocą szumu Gaussowskiego
# def mutate(individual, sigma):
#     return individual + np.random.normal(0, sigma, size=individual.shape)
def mutate_population(population, sigma):
    noiseMatrix = np.random.normal(0, sigma, (len(population), 2))
    return population + noiseMatrix

# Krzyżowanie osobników za pomocą interpolacji
def interpolation_crossover(parent1, parent2):
    a = np.random.uniform(0, 1)
    return a * parent1 + (1 - a) * parent2

# Obliczenie wartości funkcji celu dla każdego osobnika w populacji
def evaluate_population(objective_function, population):
    return [objective_function(individual[0], individual[1]) for individual in population]

# Selekcja najlepszych osobników z populacji
def select_best_individuals(population, fitness_values, num_best):
    sorted_indices = np.argsort(fitness_values)
    return [population[i] for i in sorted_indices[:num_best]]

def genetic_distance(individual1, individual2):
    return np.linalg.norm(individual1 - individual2)

def adaptive_parameters(generation, mutation_sigma):
    if generation < 50:
        mutation_sigma *= 2  # Zwiększenie mutacji na początku
    return mutation_sigma

def tournament(population, fitnessValues, mi, less_deep_minimum):
    selected_indices = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), k)
        tournament_fitness = [fitnessValues[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected_indices.append(winner_index)
    return [population[i] for i in selected_indices]

def execute_strategy(side_length, objective_function, mi, lambda_, population_size, num_generations, mutation_sigma):
    best_individuals=[]
    best_individuals_fitness=[]
    min_x = 0
    max_x = side_length
    min_y = 0
    max_y = side_length
    population = initialize_population2(population_size, min_x, min_y, max_x, max_y)
    [result1, result2] = evolutionary_strategy(objective_function, mi, lambda_, population_size, side_length, num_generations, mutation_sigma, population)
    best_individuals.append(result1)
    best_individuals_fitness.append(result2)
    min_x = -side_length
    max_x = 0
    min_y = 0
    max_y = side_length
    population = initialize_population2(population_size, min_x, min_y, max_x, max_y)
    [result1, result2] = evolutionary_strategy(objective_function, mi, lambda_, population_size, side_length, num_generations, mutation_sigma, population)
    best_individuals.append(result1)
    best_individuals_fitness.append(result2)
    min_x = -side_length
    max_x = 0
    min_y = -side_length
    max_y = 0
    population = initialize_population2(population_size, min_x, min_y, max_x, max_y)
    [result1, result2] = evolutionary_strategy(objective_function, mi, lambda_, population_size, side_length, num_generations, mutation_sigma, population)
    best_individuals.append(result1)
    best_individuals_fitness.append(result2)
    min_x = 0
    max_x = side_length
    min_y = -side_length
    max_y = 0
    population = initialize_population2(population_size, min_x, min_y, max_x, max_y)
    [result1, result2] = evolutionary_strategy(objective_function, mi, lambda_, population_size, side_length, num_generations, mutation_sigma, population)
    best_individuals.append(result1)
    best_individuals_fitness.append(result2)

    return best_individuals, best_individuals_fitness


# Strategia ewolucyjna (mi + lambda)
def evolutionary_strategy(objective_function, mi, lambda_, population_size, side_length, num_generations, mutation_sigma, population):
    # Inicjalizacja populacji

    # population = initialize_population2(population_size, side_length)
    less_deep_minimum = 0.3  # Przykładowa wartość, która określa różnicę w głębokości minimów
    for generation in range(num_generations):
        offspring = mutate_population(population, mutation_sigma)
        offspring_fitness = evaluate_population(objective_function, offspring)
        for _ in range(lambda_):
            parent1_index = np.random.choice(range(len(offspring)))
            parent2_index = np.random.choice(range(len(offspring)))
            parent1 = offspring[parent1_index]
            parent2 = offspring[parent2_index]
            child = interpolation_crossover(parent1, parent2)
            offspring = np.concatenate((offspring, [child]))
            offspring_fitness.append(objective_function(child[0], child[1]))
        
        population = select_best_individuals(offspring, offspring_fitness, mi)
        # population_fitness = evaluate_population(objective_function, population)
        # mutation_sigma = adaptive_parameters(generation, mutation_sigma)

    best_individual = max(population, key=lambda x: objective_function(x[0], x[1]))
    best_fitness = objective_function(best_individual[0], best_individual[1])
    return best_individual, best_fitness
    # population = initialize_population(population_size, side_length)
    
    # for generation in range(num_generations):
    #     # Mutacja i ewaluacja potomstwa
    #     # offspring = [mutate(individual, mutation_sigma) for individual in population for _ in range(lambda_)]
    #     offspring = mutate_population(population, mutation_sigma)
    #     offspring_fitness = evaluate_population(objective_function, offspring)

    #     for _ in range(lambda_):
    #         parent1_index = np.random.choice(range(len(offspring)))
    #         parent2_index = np.random.choice(range(len(offspring)))
    #         parent1 = offspring[parent1_index]
    #         parent2 = offspring[parent2_index]
    #         child = interpolation_crossover(parent1, parent2)
    #         offspring = np.concatenate((offspring, [child]))
    #         offspring_fitness.append(objective_function(child[0], child[1]))
        
    #     # Wybór najlepszych osobników z potomstwa
    #     # selected_offspring = select_best_individuals(offspring, offspring_fitness, mu)
            
    #     population = tournament(offspring, offspring_fitness, mi)
    #     population_fitness = evaluate_population(objective_function, population)

        
        # # Wybór najlepszych osobników z populacji rodzicielskiej
        # population_fitness = evaluate_population(objective_function, population)
        # selected_parents = select_best_individuals(population, population_fitness, mu)
        
        # # Kombinacja rodziców i potomstwa
        # population = selected_parents + selected_offspring
        
    # Zwrócenie najlepszego znalezionego osobnika
    # best_individual = max(population, key=lambda x: objective_function(x[0], x[1]))
    # best_fitness = objective_function(best_individual[0], best_individual[1])
    # return best_individual, best_fitness
    
    # return population, population_fitness