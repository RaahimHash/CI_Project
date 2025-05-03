import unfolder
import face_graph
import polytope_face_extractor
import polytope_point_generator

import numpy as np
import math
from collections import deque

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
    
    # GPT idea, but apparently dervies from Rodrigues rotation formula
    if np.allclose(n, z):  # Already aligned
        rot_matrix = np.eye(3)
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
            cur_idx2 = (cur_idx1 + 1) % len(cur.face)
            par_idx1 = par.face.index(vertex)
            par_idx2 = (par_idx1 - 1) % len(par.face)
            
            # Align first vertex
            diff = projected_faces[par.id][par_idx1] - projected_faces[cur.id][cur_idx1]
            projected_faces[cur.id] = np.array([pt + diff for pt in projected_faces[cur.id]])

            # Rotate to align second vertex - GPT Idea
            par_v1 = projected_faces[par.id][par_idx1]
            par_v2 = projected_faces[par.id][par_idx2]
            cur_v1 = projected_faces[cur.id][cur_idx1]
            cur_v2 = projected_faces[cur.id][cur_idx2]

            v1 = par_v2 - par_v1
            v2 = cur_v2 - cur_v1

            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            det = np.cross(v2_norm, v1_norm)
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
    
def visualize_flat_faces(flat_faces, face_colors=None): # visualise the flattened faces
    fig, ax = plt.subplots(figsize=(8, 8))
    patches = []
    colors = []

    for face_id, face_pts in flat_faces.items():
        polygon = Polygon(face_pts, closed=True)
        patches.append(polygon)
        if face_colors:
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

    
if __name__ == "__main__":
    points = polytope_point_generator.generate_polytope(10)
    faces, changed = polytope_face_extractor.get_conv_hull_faces(points)
    G = face_graph.make_face_graph(faces)
    faces = face_graph.fix_face_orientation(G, faces)
    T = unfolder.bfs_unfolder(G, faces)
    polygons = flatten_poly(T, points)
    visualize_flat_faces(polygons)
    face_graph.draw_dual(G)
    polytope_face_extractor.draw_polytope(points, faces, changed)
    print(faces)