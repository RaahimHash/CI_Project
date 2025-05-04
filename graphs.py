import networkx as nx
import matplotlib.pyplot as plt
import polytope_face_extractor
import polytope_point_generator
from collections import defaultdict, deque

def make_vertex_graph(faces):
    G = {}

    for face in faces:
        for v_idx in range(len(face)):
            if (face[v_idx] not in G) or (face[v_idx] in G and face[(v_idx + 1) % len(face)] not in G[face[v_idx]]): 
                G[face[v_idx]] = G.get(face[v_idx], []) + [face[(v_idx + 1) % len(face)]]
                G[face[(v_idx + 1) % len(face)]] = G.get(face[(v_idx + 1) % len(face)], []) + [face[v_idx]]

    return G   

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

def fix_face_orientation(G, faces):
    oriented = {0}
    frontier =  deque([0])
    
    while len(frontier):
        cur = frontier.popleft()
        for nei in G[cur]:
            if nei in oriented:
                continue
            
            oriented.add(nei)
            frontier.append(nei)
            for idx, vertex2 in enumerate(faces[nei]): # vertex1 is before in neighbour so it should be after in cur 
                vertex1 = faces[nei][idx-1]
                if (vertex1 in faces[cur] and vertex2 in faces[cur]):
                    vert1_pos = faces[cur].index(vertex1)
                    vert2_pos = faces[cur].index(vertex2)
                    
                    if vert2_pos == (vert1_pos + 1) % len(faces[cur]): # if vert1 is before vert2 then must flip face
                        faces[nei] = faces[nei][::-1]
                        # print("flipped")
    
    # print(len(oriented))
    return faces

def draw_vertex_graph(G):
    nxG = nx.Graph()
    for node in G:
        nxG.add_node(node)
        for neighbour in G[node]:
            if neighbour in nxG:
                nxG.add_edge(node, neighbour)
                
    pos = nx.spring_layout(nxG)
    nx.draw(nxG, pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=10)
    plt.title("Vertex Adjacency Graph")
    plt.show()

def draw_dual_graph(G):
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
    faces = fix_face_orientation(G, faces)
    draw_dual_graph(G)
    G = make_vertex_graph(faces)
    draw_vertex_graph(G)