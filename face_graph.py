import networkx as nx
import matplotlib.pyplot as plt
import polytope_face_extractor
import polytope_point_generator
from collections import defaultdict

def make_face_graph(faces):
    G = {i:[] for i in range(len(faces))}
    vertex_faces = defaultdict(set) # initialise as empty set. set because order doesn't matter, just membership does
    
    for face1_idx, face1 in enumerate(faces):  # for each face
        for vert_idx, vertex in enumerate(face1): # go through its vertices                
            # print(vert_idx, face1[vert_idx-1])
            for face2_idx in vertex_faces[vertex]: # see what other faces share my vertices
                if face2_idx in vertex_faces[face1[vert_idx-1]]: # see if the other face shared my previous vertex as well -> means edge shared -> valid for idx=0 as need to check n-1 and 0 as well
                    G[face1_idx].append(face2_idx)
                    G[face2_idx].append(face1_idx)
            
            vertex_faces[vertex].add(face1_idx) # store that i had this vertex -> useful for later faces
            
    return G

def draw_dual(G):
    nxG = nx.Graph()
    for node in G:
        nxG.add_node(node)
        for neighbour in G[node]:
            if neighbour in nxG:
                nxG.add_edge(node, neighbour)
                
    pos = nx.spring_layout(nxG)
    nx.draw(nxG, pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=10)
    plt.title("Face Adjacency Graph (Dual Graph)")
    plt.show()

if __name__=="__main__":
    points = polytope_point_generator.generate_dodec()
    faces, changed = polytope_face_extractor.get_conv_hull_faces(points)
    polytope_face_extractor.draw_polytope(points, faces, changed)
    G = make_face_graph(faces)
    draw_dual(G)