from EvolvingPopulation import EvolvingPopulation
import unfolder
import graphs
import polytope_face_extractor
import polytope_point_generator
import UnfoldingFlattener

import random
import heapq

def make_unfolder_initialiser(edge_idx):
    num_edges = len(edge_idx)
    
    def population_initialiser(population_size):
        population = []
        cur_opt = list(range(num_edges))
        for i in range(population_size):
            random.shuffle(cur_opt)
            population.append(cur_opt[:])
        return population
    return population_initialiser

def make_unfolder_fitness_and_converter(G_f, faces, points, edge_idx):
    def fitness_function(candidate, save=False, generation=0):
        T = unfolder.chromosome_to_unfolding(G_f, faces, edge_idx, candidate)
        polygons = UnfoldingFlattener.flatten_poly(T, points)
        collisions = UnfoldingFlattener.SAT(polygons)
        if save:
            UnfoldingFlattener.visualize_flat_faces(polygons, collisions, save=True, generation=generation)
        return len(faces)*len(faces) - len(collisions)
        
    def fitness_converter(fitness):
        return len(faces)*len(faces) - fitness

    return fitness_function, fitness_converter

def make_unfolder_crossover():
    def two_point_crossover_function(par1, par2):
        gene_length = len(par1) 
        crossover_point1 = random.randint(0, gene_length-1)
        crossover_point2 = (crossover_point1 + random.randint(2, gene_length - 2)) % gene_length # at least 2 genes from each parent
        child1 = [-1]*gene_length
        child2 = [-1]*gene_length
        idx = crossover_point1        
        while idx != crossover_point2:
            child1[idx] = par1[idx]            
            child2[idx] = par2[idx]
            idx = (idx + 1) % gene_length
        
        idx1 = idx
        idx2 = idx
        while idx != crossover_point1:
            while par2[idx1] in child1:
                idx1 = (idx1 + 1) % gene_length
            while par1[idx2] in child2:
                idx2 = (idx2 + 1) % gene_length
                
            child1[idx] = par2[idx1]            
            child2[idx] = par1[idx2]
            idx = (idx + 1) % gene_length
        # print("child", child1)
        return [child1, child2]
    
    # def uniform_crossover_function(par1, par2):
    #     gene_length = len(par1)
    #     child1 = []
    #     child2 = []
        
    
    return two_point_crossover_function


def make_unfolder_mutation():
    def mutation_function(candidate):
        # print("mutation", candidate)
        while random.random() < 0.9:
            gene_length = len(candidate)
            idx1 = random.randint(0, gene_length-1)
            idx2 = random.randint(0, gene_length-1)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        return candidate
    return mutation_function
        
def GeneticUnfolder(G_f, faces, points):
    edge_idx = {} # assign indexes to all edges
    for face1_idx in G_f:
        for face2_idx in G_f[face1_idx]:
            a, b = min(face1_idx, face2_idx), max(face1_idx, face2_idx) # edges always have lower face first (i, j) where i < j
            if (a, b) not in edge_idx: # if not already added
                edge_idx[(a, b)] = len(edge_idx)
    
    # shuffle edge indexes            
    indexes = list(edge_idx.values())
    random.shuffle(indexes)
    for i, edge in enumerate(list(edge_idx.keys())):
        edge_idx[edge] = indexes[i]
                
    # edge_priority = list(range(len(edge_idx)))
    # random.shuffle(edge_priority)
    # T_f = unfolder.chromosome_to_unfolding(G_f, faces, edge_idx, edge_priority)
    
    population_initialiser = make_unfolder_initialiser(edge_idx)
    fitness_function, fitness_converter = make_unfolder_fitness_and_converter(G_f, faces, points, edge_idx)
    crossover_function = make_unfolder_crossover()
    mutation_function = make_unfolder_mutation()
    
    ea_pop = EvolvingPopulation(population_initialiser=population_initialiser, population_size=100, fitness_function=fitness_function, fitness_converter=fitness_converter, crossover_function=crossover_function, num_offspring=40, mutation_function=mutation_function, mutation_rate=0.8, generations=10, preselection_func='rbs', postselection_func='trunc')
    ea_pop.evolve(verbose=True)
    return unfolder.chromosome_to_unfolding(G_f, faces, edge_idx, ea_pop.best_individual)
    
if __name__=="__main__":
    points = polytope_point_generator.generate_polytope(100)
    faces, changed = polytope_face_extractor.get_conv_hull_faces(points)
    G_f = graphs.make_face_graph(faces)
    faces = graphs.fix_face_orientation(G_f, faces)
    T_f = GeneticUnfolder(G_f)