import graphs
from polytope_point_generator import *
import polytope_face_extractor
import GeneticUnfolder
import UnfoldingFlattener
import unfolder
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import random
import time

# Modified GeneticUnfolder functions to accept face_colors
def make_unfolder_fitness_and_converter_with_colors(G_f, faces, points, edge_idx, face_colors=None):
    def fitness_function(candidate, save=False, generation=0):
        T = unfolder.chromosome_to_unfolding(G_f, faces, edge_idx, candidate)
        polygons = UnfoldingFlattener.flatten_poly(T, points)
        collisions = UnfoldingFlattener.SAT(polygons)
        if save:
            visualize_flat_faces(polygons, collisions, face_colors=face_colors, save=True, generation=generation)
        return len(faces)*len(faces) - len(collisions)
        
    def fitness_converter(fitness):
        return len(faces)*len(faces) - fitness

    return fitness_function, fitness_converter

def GeneticUnfolder_with_colors(G_f, faces, points, face_colors=None, verbose=True, collecting_data=False):
    start = time.perf_counter()
    edge_idx = {}
    for face1_idx in G_f:
        for face2_idx in G_f[face1_idx]:
            if face1_idx < face2_idx:
                edge_idx[(face1_idx, face2_idx)] = len(edge_idx)
    
    # Use the modified fitness function that accepts face_colors
    population_initialiser = GeneticUnfolder.make_unfolder_initialiser(edge_idx)
    fitness_function, fitness_converter = make_unfolder_fitness_and_converter_with_colors(G_f, faces, points, edge_idx, face_colors)
    crossover_function = GeneticUnfolder.make_unfolder_crossover()
    mutation_function = GeneticUnfolder.make_unfolder_mutation()
    
    from EvolvingPopulation import EvolvingPopulation
    
    pop_sz = 20
    ea_pop = EvolvingPopulation(
        population_initialiser=population_initialiser, 
        population_size=pop_sz, 
        fitness_function=fitness_function, 
        fitness_converter=fitness_converter, 
        crossover_function=crossover_function, 
        num_offspring=pop_sz//4, 
        mutation_function=mutation_function, 
        mutation_rate=0.9, 
        generations=2000//pop_sz, 
        preselection_func='rbs', 
        postselection_func='rbs'
    )
    ea_pop.evolve(verbose=verbose)
    end = time.perf_counter()

    if collecting_data:
        best_candidate = ea_pop.get_best_candidate()
        best_T = unfolder.chromosome_to_unfolding(G_f, faces, edge_idx, best_candidate)
        return best_T
    else:
        best_candidate = ea_pop.get_best_candidate()
        best_T = unfolder.chromosome_to_unfolding(G_f, faces, edge_idx, best_candidate)
        return best_T

# Modified draw_polytope function to use face_colors
def draw_polytope(points, faces, changed=None, only_hull_points=False, cut_edges=None, c=None, face_colors=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Turn off grid and remove all axes
    ax.grid(False)
    ax.set_axis_off()
    
    # Make panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Make pane edges invisible
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Prepare face polygons
    polygons = []
    for idx, face in enumerate(faces):
        polygon = [points[i] for i in face]
        polygons.append(polygon)

    # Use provided colors or generate pastel colors
    poly = Poly3DCollection(polygons, alpha=0.9, linewidth=0.5, edgecolor='black')
    if face_colors:
        poly.set_facecolor(face_colors)
    else:
        poly.set_facecolor('lightblue')

    ax.add_collection3d(poly)
    
    # # Add face indices at centroids
    # for idx, face in enumerate(faces):
    #     polygon = np.array([points[i] for i in face])
    #     centroid = polygon.mean(axis=0)
    #     ax.text(*centroid, str(idx), color='black', fontsize=14, ha='center', va='center')

    # Visualize the direction vector c if provided
    if c is not None:
        centroid = np.mean(points, axis=0)
        scale = np.max(np.abs(points)) * 0.5
        ax.quiver(centroid[0], centroid[1], centroid[2], c[0], c[1], c[2], 
                 color='lightblue', linewidth=3, length=scale, normalize=True, arrow_length_ratio=0.15)
        end_point = centroid + scale * c/np.linalg.norm(c)
        ax.text(end_point[0], end_point[1], end_point[2], "c", color='lightblue', fontsize=15)

    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

# Modified draw_dual_graph function to use face_colors
def draw_dual_graph(G, face_colors=None):
    nxG = nx.Graph()
    for node in G:
        nxG.add_node(node)
        for neighbour in G[node]:
            if neighbour in nxG:
                nxG.add_edge(node, neighbour)
                
    pos = nx.spring_layout(nxG)
    
    if face_colors:
        # Use the same colors for nodes as faces
        node_colors = [face_colors[node] if node < len(face_colors) else 'skyblue' for node in nxG.nodes()]
    else:
        node_colors = 'skyblue'
        
    nx.draw(nxG, pos, with_labels=True, node_color=node_colors, node_size=1000, font_size=10)
    plt.title("Face Adjacency Graph (Dual Graph)")
    plt.show()

# Modified visualize_flat_faces function to use face_colors and save option
def visualize_flat_faces(flat_faces, collisions=None, face_colors=None, save=False, generation=0):
    fig, ax = plt.subplots(figsize=(8, 8))
    patches = []
    colors = []

    if collisions:
        colliding_faces = set()
        for face1, face2 in collisions:
            colliding_faces.add(face1)
            colliding_faces.add(face2)

    for face_id, face_pts in flat_faces.items():
        polygon = Polygon(face_pts, closed=True)
        patches.append(polygon)
        
        if collisions and face_id in colliding_faces:
            colors.append('red')
        elif face_colors and face_id < len(face_colors):
            colors.append(face_colors[face_id])
        else:
            colors.append('lightblue')

        # # Label face id at centroid
        # centroid = face_pts.mean(axis=0)
        # ax.text(centroid[0], centroid[1], str(face_id), ha='center', va='center', fontsize=8)

    collection = PatchCollection(patches, facecolors=colors, edgecolor='black', alpha=0.8)
    ax.add_collection(collection)

    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.axis('off')
    plt.tight_layout()
    
    if save:
        # Create poster directory if it doesn't exist
        import os
        os.makedirs('poster', exist_ok=True)
        # Save the plot instead of showing it
        plt.savefig(f'poster/gen_{generation}.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    else:
        plt.show()

if __name__ == "__main__":
    points = generate_uniform(1000)
    # points = generate_cube()
    # points = generate_dodec()
    # points = generate_uniform(100)
    # points = generate_turtle(7, 5)
    # points = generate_flat(200)
    # points = generate_spherical(100)
    # points = generate_half_spherical(100)
    # points = generate_turtle(random.randint(1, 7), random.randint(1, 7))
    # points = generate_bumpy_turtle(random.randint(1, 7), random.randint(1, 7)) 
    # points = np.loadtxt("best_candidate_points_2.txt")

    faces, changed = polytope_face_extractor.get_conv_hull_faces(points)
    G_f = graphs.make_face_graph(faces)
    faces = graphs.fix_face_orientation(G_f, faces)
    
    # Generate consistent pastel face colors once
    face_colors = []
    for i in range(len(faces)):
        base_color = np.random.rand(3)
        base_color[0] = min(base_color[1], base_color[2])
        pastel_color = 0.6 * base_color + 0.4
        face_colors.append(tuple(pastel_color))
    
    # Use consistent colors across all visualizations
    draw_polytope(points, faces, changed, face_colors=face_colors)
    draw_dual_graph(G_f, face_colors=face_colors)
        
    # GA Unfolder with consistent colors
    T_f = GeneticUnfolder_with_colors(G_f, faces, points, face_colors=face_colors, verbose=True, collecting_data=True)
    polygons = UnfoldingFlattener.flatten_poly(T_f, points)
    count_bfs = len(polygons)
    collisions = UnfoldingFlattener.SAT(polygons)
    print("Number of collisions (GA):", len(collisions))
    visualize_flat_faces(polygons, collisions, face_colors=face_colors)
    
    # BFS Unfolder
    T_f = unfolder.bfs_unfolder(G_f, faces)
    polygons = UnfoldingFlattener.flatten_poly(T_f, points)
    count_bfs = len(polygons)
    collisions = UnfoldingFlattener.SAT(polygons)
    print("Number of collisions (BFS):", len(collisions))
    visualize_flat_faces(polygons, collisions, face_colors=face_colors)

    # Steepest Edge Unfolder
    G_v = graphs.make_vertex_graph(faces)
    collisions = 1
    while collisions:
        T_v, cut_edges, c = unfolder.steepest_edge_unfolder(G_f, faces, G_v, points) 
        polygons = UnfoldingFlattener.flatten_poly(T_v, points)
        count_steepest = len(polygons)
        collisions = UnfoldingFlattener.SAT(polygons)
        print("Number of collisions (Steepest Edge):", len(collisions))
        visualize_flat_faces(polygons, collisions, face_colors=face_colors)