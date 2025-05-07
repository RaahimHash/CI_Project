from EvolvingPopulation import EvolvingPopulation
import unfolder
import graphs
import polytope_face_extractor
import polytope_point_generator

import random
import heapq

def make_unfolder_initialiser(edge_idx):
    num_edges = len(edge_idx)
    
    def population_initialiser(population_size):
        population = []
        cur_opt = list(range(num_edges))
        for i in range(population_size):
            random.shuffle(cur_opt)
            population_size.append(cur_opt[:])
        return population
    return population_initialiser



def make_unfolder_fitness_and_converter():
    pass

def make_unfolder_crossover():
    pass

def make_unfolder_mutation():
    pass


def GeneticUnfolder(G_f, faces):
    edge_idx = {} # assign indexes to all edges
    for face1_idx in G_f:
        for face2_idx in G_f[face1_idx]:
            a, b = min(face1_idx, face2_idx), max(face1_idx, face2_idx) # edges always have lower face first (i, j) where i < j
            if (a, b) not in edge_idx: # if not already added
                edge_idx[(a, b)] = len(edge_idx)
    edge_priority = list(range(len(edge_idx)))
    random.shuffle(edge_priority)
    T_f = unfolder.chromosome_to_unfolding(G_f, faces, edge_idx, edge_priority)
    
    return T_f
    # population_initialiser = make_unfolder_initialiser(edge_idx)
    # fitness_function, fitness_converter = make_unfolder_fitness_and_converter()
    # crossover_function = make_unfolder_crossover()
    # mutation_function = make_unfolder_mutation()
    
    # ea_pop = EvolvingPopulation(population_initialiser=population_initialiser, population_size=200, fitness_function=fitness_function, fitness_converter=fitness_converter, crossover_function=crossover_function, num_offspring=120, mutation_function=mutation_function, mutation_rate=0.5, generations=10000, preselection_func='rbs', postselection_func='rbs')
    # ea_pop.evolve(verbose=True)
    
if __name__=="__main__":
    points = polytope_point_generator.generate_polytope(100)
    faces, changed = polytope_face_extractor.get_conv_hull_faces(points)
    G_f = graphs.make_face_graph(faces)
    faces = graphs.fix_face_orientation(G_f, faces)
    T_f = GeneticUnfolder(G_f)