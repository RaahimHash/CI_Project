from collections import deque
import graphs
import polytope_face_extractor
import polytope_point_generator
import numpy as np
import random
import heapq
import time

class TreeNode:
    def __init__(self, face_id, face):
        self.id = face_id
        self.face = face
        self.parent = None
        self.children = []
        
    def add_child(self, other):
        self.children.append(other)
        other.parent = self
        
    def __str__(self):
        s = f"== Diving ({self.id}) ==\n"
        for child in self.children:
            s += str(child)
        s += f"== Surfacing ({self.id}) ==\n"
        return s
    
    def __repr__(self):
        return str(self)
    
class UnfoldingTree:
    def __init__(self, root_id, root_face):
        self.root = TreeNode(root_id, root_face)

    def get_root(self):
        return self.root
    
    def __str__(self):
        return str(self.root)
    
    def __repr__(self):
        return str(self)

def bfs_unfolder(face_graph, faces, s=0):
    T = UnfoldingTree(0, faces[0])
    visited = {s}
    frontier = deque([T.get_root()])
    while len(frontier):
        cur = frontier.popleft()
        for nei in face_graph[cur.id]:
            if nei in visited:
                continue
            # print(nei) # for debugging
            visited.add(nei)
            child = TreeNode(nei, faces[nei])
            cur.add_child(child)
            frontier.append(child)    
            
    return T

def steepest_edge_unfolder(face_graph, faces, vertex_graph, points):
    
    # print("Steepest edge unfolding") # debugging
    T_v = []

    # random unit vector
    c = np.array([random.uniform(-1, 1) for _ in range(3)])
    c = c / np.linalg.norm(c) 

    # finding steepest point
    p_max = float("-inf")
    for idx in range(len(points)):
        proj = np.dot(points[idx], c)
        if proj > p_max:
            p_max = proj
            v_max = idx

    for v in vertex_graph: 
        if v == v_max: # find the most descending steepest edge from the steepest point, not in the algorithm
            steepest = float("inf")
            for nei in vertex_graph[v]:
                proj = np.dot(c, points[v] - points[nei])/np.linalg.norm(points[v] - points[nei]) 
                if proj < steepest:
                    steepest = proj
                    steepest_v = nei
            T_v.append((v, steepest_v))
        else:
            steepest = float("-inf")
            for nei in vertex_graph[v]:
                proj = np.dot(c, points[v] - points[nei])/np.linalg.norm(points[v] - points[nei]) 
                if proj > steepest:
                    steepest = proj
                    steepest_v = nei
            T_v.append((v, steepest_v))

    # print("Cut edges:", T_v) # for debugging

    T = UnfoldingTree(0, faces[0])
    visited = {0}
    parents = {0: None} # for cycle detection
    frontier = deque([T.get_root()])

    while len(frontier):
    
        cur = frontier.popleft()
        for nei in face_graph[cur.id]:

            cut = False
            intersection = set(faces[cur.id]) & set(faces[nei])
            if len(intersection) == 2:
                v1, v2 = list(intersection)
                if (v1, v2) in T_v or (v2, v1) in T_v:
                    # print("Faces are joined by a cut edge so not a neighbour")
                    cut = True
            if cut:
                continue

            if nei in visited:
                if nei != parents[cur.id]:
                    print("Cycle detected") # the dual graph is not a tree if you remove the cut edges, which means the steepest edge unfolding failed
                continue

            visited.add(nei)
            child = TreeNode(nei, faces[nei])
            cur.add_child(child)
            frontier.append(child)
            parents[nei] = cur.id

    return T, T_v, c

def chromosome_to_unfolding(G_f, faces, edge_idx, edge_priority):
    # s = time.time()
    # print("Chromosome:", edge_priority)
    # 0th face has highest priority (-1)
    heap = [(-1, 0, None)] # (priority, node, parent)
    T = None
    connected = set()
    nodes = {}
    while len(connected) < len(G_f):
        next_node = heapq.heappop(heap)
        if next_node[1] in connected: # skip nodes already in spanning tree
            continue
        if next_node[2] is None: # root
            T = UnfoldingTree(0, faces[0])
            nodes = {0: T.get_root()}
            connected = {0}
        else: # add child to spanning tree
            child = TreeNode(next_node[1], faces[next_node[1]])
            nodes[next_node[2]].add_child(child)
            # print(f"added {child.id} under parent: {next_node[2]}")
            nodes[child.id] = child
            connected.add(child.id)
        
        for nei in G_f[next_node[1]]:
            if nei in connected:
                continue
            a, b = min(next_node[1], nei), max(next_node[1], nei)
            # print(a, b)
            heapq.heappush(heap, (edge_priority[edge_idx[(a, b)]], nei, next_node[1])) # get priority from chromosome
    # e = time.time()
    # print(f"Time to generate unfolding from chromosome: {e - s}")     
    return T

if __name__ == "__main__":
    points = polytope_point_generator.generate_dodec()
    faces, changed = polytope_face_extractor.get_conv_hull_faces(points)
    # polytope_face_extractor.draw_polytope(points, faces, changed)
    G = graphs.make_face_graph(faces)
    faces = graphs.fix_face_orientation(G, faces)
    T = bfs_unfolder(G, faces)
    print(T)
    graphs.draw_dual_graph(G)