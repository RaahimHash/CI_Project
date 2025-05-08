import graphs
import polytope_point_generator
import polytope_face_extractor
import GeneticUnfolder
import UnfoldingFlattener
import unfolder

import random

if __name__ == "__main__":
    points = polytope_point_generator.generate_uniform(10000)
    # points = polytope_point_generator.generate_turtle(random.randint(1, 7), random.randint(1, 7))
    # points = polytope_point_generator.generate_flat(1000)
    # points = polytope_point_generator.generate_spherical(100)
    # points = polytope_point_generator.generate_half_spherical(40)
    # points = polytope_point_generator.generate_polytope(1000)
    # points = polytope_point_generator.generate_turtle(random.randint(1, 7), random.randint(1, 7))
    # points = polytope_point_generator.generate_dodec()
    faces, changed = polytope_face_extractor.get_conv_hull_faces(points)

    G_f = graphs.make_face_graph(faces)
    faces = graphs.fix_face_orientation(G_f, faces)
    
    # GA Unfolder
    T_f = GeneticUnfolder.GeneticUnfolder(G_f, faces, points)
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

    # # Steepest Edge Unfolder
    G_v =  graphs.make_vertex_graph(faces)
    collisions = 1
    while collisions:
        T_v, cut_edges, c = unfolder.steepest_edge_unfolder(G_f, faces, G_v, points) 
        polygons = UnfoldingFlattener.flatten_poly(T_v, points)
        count_steepest = len(polygons)
        collisions = UnfoldingFlattener.SAT(polygons)
        print("Number of collisions (Steepest Edge):", len(collisions))
        UnfoldingFlattener.visualize_flat_faces(polygons, collisions)

    # if count_bfs != count_steepest: 
    #     print("BFS and Steepest edge unfoldings have different number of faces")
    #     print("BFS:", count_bfs, "Steepest edge:", count_steepest)

    graphs.draw_dual_graph(G_f)
    polytope_face_extractor.draw_polytope(points, faces, changed)
    # polytope_face_extractor.draw_polytope(points, faces, changed, True, cut_edges, c)
    print(faces)