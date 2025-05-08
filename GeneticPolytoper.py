from EvolvingPopulation import EvolvingPopulation
import polytope_point_generator
import polytope_face_extractor
import unfolder
import graphs
import UnfoldingFlattener
import random 
import numpy as np

def make_polytoper_initialiser(pop_size, num_points, verbose = False):
    interval = (pop_size/5)/pop_size
    ranges = {"uni": (0, interval), "turtle": (interval, 2*interval), "flat": (2*interval, 3*interval), "spher": (3*interval, 4*interval), "half-spher": (4*interval, 5*interval)}

    def polytoper_initialiser(pop_size):

        if verbose:
            print("Generating population...")

        population = []
        for _ in range(pop_size):
            rand = random.random()
            population.append(polytope_point_generator.generate_uniform(num_points))

            if ranges["uni"][0] <= rand < ranges["uni"][1]:
                population.append(polytope_point_generator.generate_uniform(num_points))
    
            elif ranges["turtle"][0] <= rand < ranges["turtle"][1]:
                population.append(polytope_point_generator.generate_turtle(random.randint(1, 7), random.randint(1, 7)))

            elif ranges["flat"][0] <= rand < ranges["flat"][1]:
                population.append(polytope_point_generator.generate_flat(num_points))

            elif ranges["spher"][0] <= rand < ranges["spher"][1]:
                population.append(polytope_point_generator.generate_spherical(num_points))

            else:
                population.append(polytope_point_generator.generate_half_spherical(num_points))

        if verbose:
            print("Population generated.")
    
        return population
    
    return polytoper_initialiser

def make_polytoper_fitness_and_converter(verbose = False):
    def fitness_function(candidate, save=False, generation=0):
        
        can = candidate.copy()

        if verbose:
            print(f"Calculating fitness for {can}...")
        
        faces, changed = polytope_face_extractor.get_conv_hull_faces(can)
        G_f = graphs.make_face_graph(faces)
        faces = graphs.fix_face_orientation(G_f, faces)
        G_v = graphs.make_vertex_graph(faces)

        fitness = 0
        collisions = 1
        while collisions:
            if verbose:
                print(f"Steepest edge atteempt {fitness + 1}...")
            T_v, cut_edges, c = unfolder.steepest_edge_unfolder(G_f, faces, G_v, can) 
            if verbose:
                print(f"Steepest edge attempt {fitness + 1} done.")
            polygons = UnfoldingFlattener.flatten_poly(T_v, can)
            if verbose:
                print(f"Flattening for attempt {fitness + 1} done.")
            collisions = UnfoldingFlattener.SAT(polygons)
            if verbose:
                print(f"Number of collisions for attempt {fitness + 1}: {len(collisions)}")
            fitness += 1

        # attempt_history = []
        # for _ in range(10):
        #     attempts = 0
        #     collisions = 1
        #     while collisions:
        #         if verbose:
        #             print(f"Steepest edge atteempt {attempts + 1}...")
        #         T_v, cut_edges, c = unfolder.steepest_edge_unfolder(G_f, faces, G_v, can) 
        #         if verbose:
        #             print(f"Steepest edge attempt {attempts + 1} done.")
        #         polygons = UnfoldingFlattener.flatten_poly(T_v, can)
        #         if verbose:
        #             print(f"Flattening for attempt {attempts + 1} done.")
        #         collisions = UnfoldingFlattener.SAT(polygons)
        #         if verbose:
        #             print(f"Number of collisions for attempt {attempts + 1}: {len(collisions)}")
        #         attempts += 1
        #     attempt_history.append(attempts)

        # fitness = np.mean(attempt_history)
        
        if verbose:
            print(f"Fitness for {can}: {fitness}")

        return fitness 
    def fitness_converter(fitness):
        if verbose:
            print(f"Converting fitness: {fitness}")
        return fitness
    return fitness_function, fitness_converter

def make_polytoper_crossover(num_points, verbose = False):

    def crossover_function(par1, par2):
        
        if verbose:
            print(f"Crossing over {par1} and {par2}...")
        
        child1 = []
        child2 = []

        p1_idx = 0
        p2_idx = 0
        while len(child1) < num_points:
            rand = random.random()
            if rand <= 0.5:
                child1.append(par1[p1_idx])
                p1_idx = (p1_idx + 1) % len(par1)
                child2.append(par2[p2_idx])
                p2_idx = (p2_idx + 1) % len(par2)
            else:
                child1.append(par2[p2_idx])
                p2_idx = (p2_idx + 1) % len(par2)
                child2.append(par1[p1_idx])
                p1_idx = (p1_idx + 1) % len(par1)

        child1 = np.array(child1)
        child1 = np.unique(child1, axis=0)
        child2 = np.array(child2)
        child2 = np.unique(child2, axis=0)

        if verbose:
            print(f"Children generated: {child1}, {child2}")

        return [child1, child2]

    return crossover_function

def make_polytoper_mutation(percentage, verbose = False):

    def mutation_function(candidate):

        if verbose:
            print(f"Mutating {candidate}...")
        
        start_idx = random.randint(0, len(candidate) - int(len(candidate)*percentage))
        end_idx = random.randint(start_idx, start_idx + int(len(candidate)*percentage))

        min_x = min(candidate[:, 0])
        max_x = max(candidate[:, 0])
        min_y = min(candidate[:, 1])
        max_y = max(candidate[:, 1])
        min_z = min(candidate[:, 2])
        max_z = max(candidate[:, 2])

        for i in range(start_idx, end_idx):

            candidate[i, 0] = random.uniform(min_x, max_x)
            candidate[i, 1] = random.uniform(min_y, max_y)
            candidate[i, 2] = random.uniform(min_z, max_z)

        candidate = np.unique(candidate, axis=0)

        if verbose:
            print(f"Mutated candidate: {candidate}")

        return candidate

    return mutation_function

if __name__ == "__main__":
    
    pop_size = 100
    num_points = 1000
    population_initialiser = make_polytoper_initialiser(pop_size, num_points, verbose = False)
    fitness_function, fitness_converter = make_polytoper_fitness_and_converter(verbose = False)
    crossover_function = make_polytoper_crossover(num_points, verbose = False)
    mutation_function = make_polytoper_mutation(0.1, verbose = False)
    
    ea_pop = EvolvingPopulation(population_initialiser=population_initialiser, population_size=pop_size, fitness_function=fitness_function, fitness_converter=fitness_converter, crossover_function=crossover_function, num_offspring=60, mutation_function=mutation_function, mutation_rate=0.5, generations=100, preselection_func='bin_tour', postselection_func='bin_tour', elitism=True, recompute=False)

    ea_pop.evolve(verbose=True)

    np.savetxt("best_candidate_points.txt", ea_pop.best_individual, fmt='%.10f')
    with open("best_candidate_info.txt", "w") as f:
        f.write(f"Fitness: {ea_pop.best_fitness}\n")

    # verify
    points = ea_pop.best_individual
    faces, changed = polytope_face_extractor.get_conv_hull_faces(points)
    G_f = graphs.make_face_graph(faces)
    faces = graphs.fix_face_orientation(G_f, faces)
    G_v = graphs.make_vertex_graph(faces)
    T_v, cut_edges, c = unfolder.steepest_edge_unfolder(G_f, faces, G_v, points)
    polygons = UnfoldingFlattener.flatten_poly(T_v, points)
    collisions = UnfoldingFlattener.SAT(polygons)
    print("Number of collisions (best candidate):", len(collisions))
    UnfoldingFlattener.visualize_flat_faces(polygons, collisions)
    polytope_face_extractor.draw_polytope(points, faces, changed)