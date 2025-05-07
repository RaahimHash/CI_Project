import unfolder
import graphs
import polytope_face_extractor
import polytope_point_generator

import numpy as np
import math
import random
from collections import deque
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import numpy as np
from scipy.spatial.transform import Rotation as R

def project_face_by_normal(points, face): # project face onto xy-plane
    # Get three points from the face
    p0, p1, p2 = [points[i] for i in face[:3]]
    v1 = p1 - p0
    v2 = p2 - p0
    n = np.cross(v1, v2)
    n = n / np.linalg.norm(n)

    # Rotation from normal to Z-axis
    z = np.array([0, 0, 1])
    
    # GPT idea, but apparently derives from Rodrigues rotation formula
    if np.allclose(n, z):  # Already aligned
        rot_matrix = np.eye(3) # eye-dentity matrix lmao
    elif np.allclose(n, -z):  # 180-degree rotation
        rot_matrix = R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
    else:
        axis = np.cross(n, z)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(n, z), -1.0, 1.0))
        rot_matrix = R.from_rotvec(axis * angle).as_matrix()
    
    # Apply rotation to the face, using p0 as the origin
    face_points = np.array([points[i] - p0 for i in face])  # translate to origin
    rotated = face_points @ rot_matrix.T  # apply rotation
    projected = rotated[:, :2]  # drop z
    return projected

def rotate_2d(points, center, angle): # for aligning projected polygons
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    return (points - center) @ rot_matrix.T + center

def align_to_parent(projected_faces, cur):
    par = cur.parent
    for cur_idx1, vertex in enumerate(cur.face):
        if vertex in par.face:
            if cur_idx1 == 0 and cur.face[-1] in par.face:
                cur_idx1 = len(cur.face)-1
                vertex = cur.face[cur_idx1]
                
            cur_idx2 = (cur_idx1 + 1) % len(cur.face)
            par_idx1 = par.face.index(vertex)
            par_idx2 = (par_idx1 - 1) % len(par.face)
            # print("par, cur", par.face[par_idx2], cur.face[cur_idx2])
            
            # Align first vertex
            diff = projected_faces[par.id][par_idx1] - projected_faces[cur.id][cur_idx1]
            projected_faces[cur.id] = np.array([pt + diff for pt in projected_faces[cur.id]])

            # Rotate to align second vertex - GPT Idea
            par_v1 = projected_faces[par.id][par_idx1]
            par_v2 = projected_faces[par.id][par_idx2]
            cur_v1 = projected_faces[cur.id][cur_idx1]
            cur_v2 = projected_faces[cur.id][cur_idx2]

            v1 = par_v1 - par_v2
            v2 = cur_v1 - cur_v2

            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            det = np.cross(v2_norm, v1_norm)
            # print(det, dot)
            angle = np.arctan2(det, dot)

            # Rotate around par_v1
            projected_faces[cur.id] = rotate_2d(projected_faces[cur.id], par_v1, angle)
            break  # found shared edge

def flatten_poly(T: unfolder.UnfoldingTree, points):
    root = T.get_root()
    frontier = deque([root])
    projected_faces = {}
    while len(frontier):
        cur = frontier.popleft()
        projected_faces[cur.id] = project_face_by_normal(points, cur.face) # project to xy (maintain proportion)
        
        if cur.parent: # align to parent
            align_to_parent(projected_faces, cur)
            
        for child in cur.children: # need to unfold my children as well
            frontier.append(child)
            
    return projected_faces
    
def visualize_flat_faces(flat_faces, collisions=None, face_colors=None): # visualise the flattened faces
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
        elif face_colors:
            colors.append(face_colors.get(face_id, 'lightblue'))
        else:
            colors.append('lightblue')

        # Optionally label face id at centroid
        centroid = face_pts.mean(axis=0)
        ax.text(centroid[0], centroid[1], str(face_id), ha='center', va='center', fontsize=8)

    collection = PatchCollection(patches, facecolors=colors, edgecolor='black', alpha=0.8)
    ax.add_collection(collection)

    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def SAT(flat_faces): # no broad phase check, just narrow phase check
    face_ids = list(flat_faces.keys())
    colliding_faces = set()

    for i in range(len(face_ids)-1):
        for j in range(i+1, len(face_ids)):
            poly1 = flat_faces[face_ids[i]]
            poly2 = flat_faces[face_ids[j]]
            colliding = True
            
            axes = []
            for p in range(len(poly1)):
                p1 = poly1[p]
                p2 = poly1[(p + 1) % len(poly1)]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                axes.append(normal / np.linalg.norm(normal))
            for p in range(len(poly2)):
                p1 = poly2[p]
                p2 = poly2[(p + 1) % len(poly2)]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                axes.append(normal / np.linalg.norm(normal))

            for axis in axes:
                # Copilot 
                min1, max1 = float('inf'), float('-inf')
                min2, max2 = float('inf'), float('-inf')
                for p in poly1:
                    projection = np.dot(p, axis)
                    min1 = min(min1, projection)
                    max1 = max(max1, projection)
                for p in poly2:
                    projection = np.dot(p, axis)
                    min2 = min(min2, projection)
                    max2 = max(max2, projection)
                if max1 < min2 or np.isclose(max1, min2) or max2 < min1 or np.isclose(max2, min1):
                    colliding = False 
                    break

            if colliding:
                colliding_faces.add((face_ids[i], face_ids[j]))

    return colliding_faces
    
if __name__ == "__main__":
    # points = polytope_point_generator.generate_uniform(10000)
    # points = polytope_point_generator.generate_turtle(random.randint(1, 7), random.randint(1, 7))
    points = polytope_point_generator.generate_flat(1000)
    # points = polytope_point_generator.generate_spherical(100)
    # points = polytope_point_generator.generate_half_spherical(100)
    faces, changed = polytope_face_extractor.get_conv_hull_faces(points)

    G_f = graphs.make_face_graph(faces)
    faces = graphs.fix_face_orientation(G_f, faces)
    T_f = unfolder.bfs_unfolder(G_f, faces)
    polygons = flatten_poly(T_f, points)
    count_bfs = len(polygons)
    collisions = SAT(polygons)
    print("Number of collisions:", len(collisions))
    visualize_flat_faces(polygons, collisions)
    
    while True:
        G_v =  graphs.make_vertex_graph(faces)
        T_v, cut_edges, c = unfolder.steepest_edge_unfolder(G_f, faces, G_v, points) 
        polygons = flatten_poly(T_v, points)
        count_steepest = len(polygons)
        collisions = SAT(polygons)
        print("Number of collisions:", len(collisions))
        visualize_flat_faces(polygons, collisions)
        if len(collisions) == 0:
            break

    if count_bfs != count_steepest: 
        print("BFS and Steepest edge unfoldings have different number of faces")
        print("BFS:", count_bfs, "Steepest edge:", count_steepest)

    # graphs.draw_dual_graph(G_f)
    # polytope_face_extractor.draw_polytope(points, faces, changed)
    polytope_face_extractor.draw_polytope(points, faces, changed, True, cut_edges, c)
    # print(faces)