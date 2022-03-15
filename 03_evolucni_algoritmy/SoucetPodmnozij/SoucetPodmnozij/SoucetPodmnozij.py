import numpy as np
import random
import copy
import matplotlib.pyplot as plt
print_stats = False



def random_population(population_size, individual_size):
    population = []
    
    for i in range(0,population_size):
        individual = np.random.choice([0, 1], size=(individual_size,), p=[1/2, 1/2])
        population.append(individual)
        
    return population

#Upravit
def fitness(individual):
    sum = 0
    for k in range(len(individual)):
        sum += individual[k]*random_set[k]
    if (value - sum == 0):
        return 0
    else:
        return 1/abs(value - sum)


#Roulette selection
def selection(population,fitness_value):
    return copy.deepcopy(random.choices(population, weights=fitness_value, k=len(population))) 

def crossover(population,cross_prob=1):
    new_population = []
    
    for i in range(0,len(population)//2):
        indiv1 = copy.deepcopy(population[2*i])
        indiv2 = copy.deepcopy(population[2*i+1])
        
        if print_stats:
            print(f'Mom: {indiv1}')
            print(f'Dad: {indiv2}')
            
        if random.random()<cross_prob:
            # zvolime index krizeni nahodne
            crossover_point = random.randint(0, len(indiv1)) 
            end2 =  copy.deepcopy(indiv2[:crossover_point])
            indiv2[:crossover_point] = indiv1[:crossover_point]
            indiv1[:crossover_point] = end2
            
            if print_stats:
                print(f'Crossover point: {crossover_point}' )
                print(f'Son: {indiv1}' )
                print(f'Daughter: {indiv2}' )
                print(f'----------')

        new_population.append(indiv1)
        new_population.append(indiv2)
        
    return new_population

def mutation(population,indiv_mutation_prob=0.1,bit_mutation_prob=0.2):
    new_population = []
    
    for i in range(0,len(population)):
        individual = copy.deepcopy(population[i])
        if random.random() < indiv_mutation_prob:
            for j in range(0,len(individual)):
                if random.random() < bit_mutation_prob:
                    if individual[j]==1:
                        individual[j] = 0
                    else:
                        individual[j] = 1
                        
                    if print_stats:
                        print(f'Mutated bit {j} in the individual {i} to value {individual}')
                        
        new_population.append(individual)
        
    return new_population



def evolution(population_size, individual_size, max_generations):
    max_fitness = []
    population = random_population(population_size,individual_size)
    
    for i in range(0,max_generations):
        fitness_value = list(map(fitness, population))
        max_fitness.append(max(fitness_value))
        parents = selection(population,fitness_value)
        children = crossover(parents)
        mutated_children = mutation(children)
        population = mutated_children
        
    # spocitame fitness i pro posledni populaci
    fitness_value = list(map(fitness, population))
    max_fitness.append(max(fitness_value))
    best_individual = population[np.argmax(fitness_value)]
    
    return best_individual, population, max_fitness
 
    
# for cyklus vvyse se da napsat i v jednom prikazu
random_set = [random.randint(0,250) for i in range(0,500)]
    
value = sum(random_set)//2
print(value, random_set)

best, population, max_fitness = evolution(population_size=500,individual_size=500,max_generations=10)

print('best fitness: ', fitness(best))
print('best individual: ', best)

sum = 0
for k in range(len(best)):
        sum += best[k]*random_set[k]
print(sum)


plt.plot(max_fitness)
plt.ylabel('Fitness')
plt.xlabel('Generace')








