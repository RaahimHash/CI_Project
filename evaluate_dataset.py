import graphs
from polytope_point_generator import *
import polytope_face_extractor
import GeneticUnfolder
import UnfoldingFlattener
import unfolder
import numpy as np
import os
import csv

NUM_SHAPES = 20

# datasets = ["unif5k"]
# datasets = ["turt"]
datasets = ["unif2k", "flat2k"]
# datasets = ["half_spher", "spher"]
# datasets[0] = "flat2k"

for dataset in datasets:
    os.makedirs(f"results/{dataset}", exist_ok=True)

    # for table data, not to use directly
    num_faces = {p: 0 for p in range(NUM_SHAPES)} # number of faces for each shape
    unfolded_within = {20: 0, 40: 0, 60: 0, 80: 0, 100: 0}
    unfolded_by_bfs = 0
    avg_attempts = {p:0 for p in range(NUM_SHAPES)} # average of attempts to unfold by steepest edge
    time_taken = {p: 0 for p in range(NUM_SHAPES)} # time taken for each shape

    # plot
    avg_fitnesses = {p:[] for p in range(NUM_SHAPES)} # avg fitness of each shape
    best_fitnesses = {p:[] for p in range(NUM_SHAPES)} # best fitness of each shape

    for p in range(NUM_SHAPES):
        print(f"Processing {dataset} shape {p+1}/{NUM_SHAPES}")

        points = np.loadtxt(f"temp_dataset/{dataset}/{p}.txt")

        faces, changed = polytope_face_extractor.get_conv_hull_faces(points)
        num_faces[p] = len(faces)
        G_f = graphs.make_face_graph(faces)
        faces = graphs.fix_face_orientation(G_f, faces)

        # GA unfolder
        T_f, time_taken[p], best_fitnesses[p], avg_fitnesses[p] = GeneticUnfolder.GeneticUnfolder(G_f, faces, points, verbose=False, collecting_data=True)
        print(f"GA finished for shape {p+1}/{NUM_SHAPES}")
        for t, collisions in enumerate(best_fitnesses[p]):
            if collisions == 0:
                for within in unfolded_within:
                    if t+1 <= within:
                        unfolded_within[within] += 1
                break 

        # BFS Unfolder 
        T_f = unfolder.bfs_unfolder(G_f, faces)
        polygons = UnfoldingFlattener.flatten_poly(T_f, points)
        collisions = UnfoldingFlattener.SAT(polygons)
        if len(collisions) == 0:
            unfolded_by_bfs += 1
        
        # Steepest Edge Unfolder
        G_v =  graphs.make_vertex_graph(faces)
        attempts = []

        for _ in range(10):
            count = 0
            collisions = 1
            while collisions:
                T_v, cut_edges, c = unfolder.steepest_edge_unfolder(G_f, faces, G_v, points) 
                polygons = UnfoldingFlattener.flatten_poly(T_v, points)
                collisions = UnfoldingFlattener.SAT(polygons)
                count += 1
            attempts.append(count)

        avg_attempts[p] = np.mean(attempts)

    # actual table data
    avg_num_faces = np.mean(list(num_faces.values()))
    percent_unfolded_within = {within: (unfolded_within[within] / NUM_SHAPES) * 100 for within in unfolded_within}
    percent_unfolded_by_bfs = (unfolded_by_bfs / NUM_SHAPES) * 100
    avg_avg_attempts = np.mean(list(avg_attempts.values()))
    avg_time_taken = np.mean(list(time_taken.values()))

    with open(f"results/{dataset}/table.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Category', 'Avg Faces', '% Unfolded (20 Gen)', 
                        '% Unfolded (40 Gen)', '% Unfolded (60 Gen)', 
                        '% Unfolded (80 Gen)', '% Unfolded (100 Gen)',
                        '% Unfolded by BFS', 'Avg Attempts (Steepest)', 
                        'Avg Time (sec)'])

    # Inside your dataset loop, after calculating all metrics:
    # Write the row for this dataset
    with open(f"results/{dataset}/table.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            dataset,
            f"{avg_num_faces:.2f}",
            f"{percent_unfolded_within[20]:.1f}",
            f"{percent_unfolded_within[40]:.1f}",
            f"{percent_unfolded_within[60]:.1f}",
            f"{percent_unfolded_within[80]:.1f}",
            f"{percent_unfolded_within[100]:.1f}",
            f"{percent_unfolded_by_bfs:.1f}",
            f"{avg_avg_attempts:.2f}",
            f"{avg_time_taken:.2f}"
        ])
    
    # Save the detailed fitness data in a retrievable format
    # Best fitness for each shape at each generation
    best_fitness_array = np.array([best_fitnesses[p] for p in range(NUM_SHAPES)])
    np.savetxt(f"results/{dataset}/best_fitnesses.txt", best_fitness_array)
    
    # Average fitness for each shape at each generation
    avg_fitness_array = np.array([avg_fitnesses[p] for p in range(NUM_SHAPES)])
    np.savetxt(f"results/{dataset}/avg_fitnesses.txt", avg_fitness_array)
    
    # Detailed raw data for advanced analysis
    with open(f"results/{dataset}/raw_data.json", 'w') as f:
        import json
        json.dump({
            'num_faces': num_faces,
            'best_fitnesses': {str(p): best_fitnesses[p] for p in range(NUM_SHAPES)},
            'avg_fitnesses': {str(p): avg_fitnesses[p] for p in range(NUM_SHAPES)},
            'time_taken': time_taken,
            'avg_attempts': avg_attempts
        }, f)
    
    print(f"Processed {dataset}: {percent_unfolded_within[100]:.1f}% unfolded within 100 generations")