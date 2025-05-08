import graphs
from polytope_point_generator import *
import polytope_face_extractor
import GeneticUnfolder
import UnfoldingFlattener
import unfolder
import numpy as np

import random

if __name__ == "__main__":
    # points = generate_uniform(5000)
    # points = generate_turtle(random.randint(1, 7), random.randint(1, 7), COUNT = 10)
    # points = generate_flat(5000)
    # points = generate_spherical(100)
    # points = generate_half_spherical(100)
    # points = generate_turtle(random.randint(1, 7), random.randint(1, 7))
    # points = generate_dodec()
    # points = generate_bumpy_turtle(random.randint(1, 7), random.randint(1, 7)) 
    points = np.loadtxt("best_candidate_points.txt")
    # points = generate_cube()


    # all_points = [("Uniform 1", generate_uniform(5000)), ("Uniform 2", generate_uniform(2000)), ("Uniform 3", generate_uniform(500)), ("Flat 1", generate_flat(5000)), ("Flat 2", generate_flat(2000)), ("Turtle", generate_turtle(5, 5)), ("Half-Spherical", generate_half_spherical(50))]

    # for case, points in all_points:
    #     print(f"Case: {case} - Faces = {len(faces)}")

    faces, changed = polytope_face_extractor.get_conv_hull_faces(points)
    G_f = graphs.make_face_graph(faces)
    faces = graphs.fix_face_orientation(G_f, faces)
    polytope_face_extractor.draw_polytope(points, faces, changed)
    graphs.draw_dual_graph(G_f)
        
    # GA Unfolder
    T_f = GeneticUnfolder.GeneticUnfolder(G_f, faces, points, verbose=True)
    polygons = UnfoldingFlattener.flatten_poly(T_f, points)
    count_bfs = len(polygons)
    collisions = UnfoldingFlattener.SAT(polygons)
    print("Number of collisions (GA):", len(collisions))
    UnfoldingFlattener.visualize_flat_faces(polygons, collisions)
    
    # BFS Unfolder
    T_f = unfolder.bfs_unfolder(G_f, faces)
    polygons = UnfoldingFlattener.flatten_poly(T_f, points)
    count_bfs = len(polygons)
    collisions = UnfoldingFlattener.SAT(polygons)
    print("Number of collisions (BFS):", len(collisions))
    UnfoldingFlattener.visualize_flat_faces(polygons, collisions)

    # Steepest Edge Unfolder
    G_v =  graphs.make_vertex_graph(faces)
    collisions = 1
    while collisions:
        T_v, cut_edges, c = unfolder.steepest_edge_unfolder(G_f, faces, G_v, points) 
        polygons = UnfoldingFlattener.flatten_poly(T_v, points)
        count_steepest = len(polygons)
        collisions = UnfoldingFlattener.SAT(polygons)
        print("Number of collisions (Steepest Edge):", len(collisions))
        UnfoldingFlattener.visualize_flat_faces(polygons, collisions)

    # # if count_bfs != count_steepest: 
    # #     print("BFS and Steepest edge unfoldings have different number of faces")
    # #     print("BFS:", count_bfs, "Steepest edge:", count_steepest)

    # graphs.draw_dual_graph(G_f)
    # polytope_face_extractor.draw_polytope(points, faces, changed)
    # polytope_face_extractor.draw_polytope(points, faces, changed, True, cut_edges, c)
    print(faces)