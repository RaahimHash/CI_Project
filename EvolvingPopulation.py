import random
import matplotlib.pyplot as plt

class EvolvingPopulation():
    '''
    This class creates objects of evolutionary algorithms. Evolve function carries out the general procedure of evolution.
    Creating an EA requires:
        - Population Initialiser function (scenario specific - e.g. random permutation) - a function which takes in population size and returns a population
        - Population Size (e.g. 30)
        - Fitness function (scenario specific - e.g. total time for schedule) - a function which takes in a candidate and returns its fitness
        - Fitness converter (scenario specific) - a function which takes the fitness value (which is to be maximised) and converts it into an appropriate output for the problem (a quantity which may need to be minimized like total time in JSSP)
        - Crossover function (scenario specific - e.g. single crossover point while ensuring no repeats) - a function which takes a pair of parents and returns children (without mutation)
        - Number of Offspring (e.g. 10)
        - Mutation function (scenario specific - e.g. swap gene) - a function that takes a candidate and returns a mutated version of the child
        - Mutation rate (e.g. 0.5)
        - Number of Generations to simulate (e.g. 50)
        - Name of Parent/Preselection function ('fps', 'rbs', 'bin_tour', 'trunc', 'rand')
        - Name of Survivor/Postselection function ('fps', 'rbs', 'bin_tour', 'trunc', 'rand')
    '''
    
    # EA Parameter Setup
    def __init__(self, population_initialiser, population_size, fitness_function, fitness_converter, crossover_function, num_offspring, mutation_function, mutation_rate, generations, preselection_func, postselection_func, elitism=False, recompute=False):
        self.population_size = population_size
        self.population = population_initialiser(self.population_size)
        self.fitness_function = fitness_function
        self.fitness = [self.fitness_function(candidate) for candidate in self.population]
        self.fitness_converter = fitness_converter
        self.generations = generations
        self.elitism = elitism
        self.recompute = recompute
        
        self.pre_selection_function = EvolvingPopulation.selection_functions[preselection_func]
        self.post_selection_function = EvolvingPopulation.selection_functions[postselection_func]
        
        self.crossover_function = crossover_function
        self.num_offspring = num_offspring
        self.mutation_function = mutation_function
        self.mutation_rate = mutation_rate
        
        self.best_individual = None
        self.best_fitness = -999999
        
        self.best_fitness_history = [] # 0 idx is after 1st crossover
        self.mean_fitness_history = []
    
    # Evolution    
    def evolve(self, verbose=False):
        for gen in range(self.generations):
            # Expand
            # print("expand")
            parents, _ = self.pre_selection_function(self.population, self.fitness, selection_sz=self.num_offspring) # select parents
            random.shuffle(parents) # to ensure pairings are random
            pair_parents = [(parents[i], parents[i-1]) for i in range(1, len(parents), 2)] # make parent pairings
            children = []
            for par1, par2 in pair_parents:
                for child in self.crossover_function(par1, par2): # create children
                    while random.random() < self.mutation_rate: # mutate based on chance
                        child = self.mutation_function(child)
                    children.append(child) # store generated child
            self.population.extend(children) # add children to population
            if self.recompute: 
                self.fitness = [self.fitness_function(candidate) for candidate in self.population] 
            else:
                self.fitness.extend([self.fitness_function(child) for child in children])
                
            
            # Contract
            # print("contract")
            # self.fitness = [self.fitness_function(candidate) for candidate in self.population] # re-evaluate fitness
            if self.elitism and self.best_individual is not None:  
                self.population, self.fitness = self.post_selection_function(self.population, self.fitness, selection_sz=self.population_size-1) # survivor selection
                self.population.append(self.best_individual)
                self.fitness.append(self.best_fitness)
            else:
                self.population, self.fitness = self.post_selection_function(self.population, self.fitness, selection_sz=self.population_size) # survivor selection
            
            # Analyse
            # print("analyse")
            cur_max = max(self.fitness)
            if cur_max > self.best_fitness: # update best
                self.best_fitness = cur_max
                self.best_individual = self.population[self.fitness.index(self.best_fitness)].copy()
                # if gen % 10 == 0:
                try:
                    self.fitness_function(self.best_individual, save=True, generation=gen)
                except:
                    pass
                    # print("generation failed")
            self.best_fitness_history.append(self.fitness_converter(cur_max))
            self.mean_fitness_history.append(self.fitness_converter(sum(self.fitness)/len(self.fitness)))
            
                
            if verbose:
                print(f"Best till Gen {gen}: {self.best_fitness_history[-1]}")
                print(f"Average current fitness in Gen {gen}: {self.mean_fitness_history[-1]}")
                
    def plot_results(self):
        plt.plot(list(range(self.generations)), self.best_fitness_history, label='Best Fitness')
        plt.plot(list(range(self.generations)), self.mean_fitness_history, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness over Generations')
        plt.legend()
        plt.show()
                            
    
    # Selection functions
    def univ_random_sample(population, fitness, proportion, selection_sz):
        pop_sz = len(population)
        sigma = random.random()*(1/selection_sz)
        selected = []
        selected_fitness = []
        
        i = 0
        cur = proportion[0]
        while i < pop_sz and sigma < 1:
            if sigma < cur:
                selected.append(population[i])
                selected_fitness.append(fitness[i])
                sigma += 1/selection_sz
            else:
                i += 1
                cur += proportion[i]
        return selected, selected_fitness
    
    def fitness_proportional(population, fitness, selection_sz):
        total_fitness = sum(fitness)
        fitness_prop = [fit/total_fitness for fit in fitness]
        return EvolvingPopulation.univ_random_sample(population, fitness, fitness_prop, selection_sz)
    
    def rank_based(population, fitness, selection_sz):
        fitness_index = list(zip(fitness, range(len(fitness)))) # maintain index
        fitness_index.sort() # lowest fitness first
        
        index_rank = [] # lower rank is for lower fitness num
        total_ranks = 0
        for rank, fit_ind in enumerate(fitness_index):
            index_rank.append((fit_ind[1], rank+1))
            total_ranks += rank+1
        index_rank.sort()
        
        rank_prop = [ind_rank[1]/total_ranks for ind_rank in index_rank]
        return EvolvingPopulation.univ_random_sample(population, fitness, rank_prop, selection_sz)     
    
    def random_selection(population, fitness, selection_sz):
        return EvolvingPopulation.univ_random_sample(population, fitness, [1/len(population)]*len(population), selection_sz)
    
    def binary_tournament(population, fitness, selection_sz):
        selected = []
        selected_fitness = []
        for i in range(selection_sz):
            # select two for tournament
            opt1 = random.randint(0, len(population) - 1)
            opt2 = random.randint(0, len(population) - 1)
            if fitness[opt2] > fitness[opt1]: # set the better to opt1
                opt1 = opt2
                
            # select opt1 (which is the better of the two)
            selected.append(population[opt1])
            selected_fitness.append(fitness[opt1])
        return selected, selected_fitness
            
    
    def truncation(population, fitness, selection_sz):
        fit_pop = list(zip(fitness, population))
        
        fit_pop.sort(reverse=True)
        fit_pop = fit_pop[:selection_sz]
        fit = [indiv[0] for indiv in fit_pop]
        pop = [indiv[1] for indiv in fit_pop]
        return pop, fit
    
    selection_functions = {'fps': fitness_proportional, 'rbs': rank_based, 'bin_tour': binary_tournament, 'trunc': truncation, 'rand': random_selection, }

