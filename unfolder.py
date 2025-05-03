from collections import deque
import face_graph
import polytope_face_extractor
import polytope_point_generator

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
    
    
if __name__ == "__main__":
    points = polytope_point_generator.generate_dodec()
    faces, changed = polytope_face_extractor.get_conv_hull_faces(points)
    # polytope_face_extractor.draw_polytope(points, faces, changed)
    G = face_graph.make_face_graph(faces)
    faces = face_graph.fix_face_orientation(G, faces)
    T = bfs_unfolder(G, faces)
    print(T)
    face_graph.draw_dual(G)