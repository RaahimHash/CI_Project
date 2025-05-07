from scipy.spatial import ConvexHull
import numpy as np
import random 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from polytope_point_generator import *

# def check_if_planar(i, j, k, p):

#     A = np.array([i, j, k, p])
#     det = np.linalg.det(A)
    
#     return det == 0``

# # for comparing the two methods of ordering vertices
# points = np.array([
#     [0.0, 0.0, -100.0], [0.0, 0.0, -100.0], [0.0, 0.0, -100.0], [0.0, 0.0, -100.0], [0.0, 0.0, -100.0], [0.0, 0.0, -100.0], [0.0, 0.0, -100.0], [19.0, 7.0, 77.0], [19.0, 7.0, -100.0], [19.0, 7.0, -100.0], [19.0, 7.0, -100.0], [19.0, 7.0, -100.0], [19.0, 7.0, -100.0], [53.0, 37.0, -52.0], [53.0, 37.0, -100.0], [84.0, -14.0, -62.0], [84.0, -14.0, -100.0], [84.0, -14.0, -100.0], [84.0, -14.0, -100.0], [87.0, 4.0, -65.0], [87.0, 4.0, -100.0], [87.0, 4.0, -100.0], [87.0, 4.0, -100.0], [87.0, 4.0, -100.0], [87.0, 4.0, -100.0], [87.0, 4.0, -100.0], [87.0, 4.0, -100.0], [87.0, 4.0, -100.0], [87.0, 4.0, -100.0], [87.0, 4.0, -100.0], [87.0, 4.0, -100.0], [87.0, 4.0, -100.0], [87.0, 4.0, -100.0], [-54.0, -81.0, -78.0], [-54.0, -81.0, -100.0], [-54.0, -81.0, -100.0], [-86.0, -99.0, 50.0], [-47.0, 34.0, 63.0], [-47.0, 34.0, -100.0], [-47.0, 34.0, -100.0], [-47.0, 34.0, -100.0], [33.0, 98.0, 84.0], [33.0, 98.0, -100.0], [33.0, 98.0, -100.0], [33.0, 98.0, -100.0], [33.0, 80.0, -61.0], [33.0, 80.0, -100.0], [33.0, 80.0, -100.0], [33.0, 80.0, -100.0], [33.0, 80.0, -100.0], [33.0, 80.0, -100.0], [-62.0, -23.0, 5.0], [-62.0, -23.0, -100.0], [-62.0, -23.0, -100.0], [-62.0, -23.0, -100.0], [-62.0, -23.0, -100.0], [-62.0, -23.0, -100.0], [-38.0, -73.0, -20.0], [-38.0, -73.0, -100.0], [-38.0, -73.0, -100.0], [-38.0, -73.0, -100.0], [-38.0, -73.0, -100.0], [-38.0, -73.0, -100.0], [-38.0, -73.0, -100.0], [-38.0, -73.0, -100.0], [-75.0, -39.0, -18.0], [-75.0, -39.0, -100.0], [-75.0, -39.0, -100.0], [-75.0, -39.0, -100.0], [-75.0, -39.0, -100.0], [-75.0, -39.0, -100.0], [59.0, -64.0, -8.0], [59.0, -64.0, -100.0], [59.0, -64.0, -100.0], [59.0, -64.0, -100.0], [59.0, -64.0, -100.0], [59.0, -64.0, -100.0], [-18.0, -35.0, -54.0], [-18.0, -35.0, -100.0], [-18.0, -35.0, -100.0], [81.0, -17.0, -75.0], [81.0, -17.0, -100.0], [81.0, -17.0, -100.0], [81.0, -17.0, -100.0], [-29.0, -2.0, 6.0], [-29.0, -2.0, -100.0], [-29.0, -2.0, -100.0], [-29.0, -2.0, -100.0], [-29.0, -2.0, -100.0], [-29.0, -2.0, -100.0], [-29.0, -2.0, -100.0], [-29.0, -2.0, -100.0], [-29.0, -2.0, -100.0], [-55.0, -95.0, -58.0], [-55.0, -95.0, -100.0], [-55.0, -95.0, -100.0], [-55.0, -95.0, -100.0], [-13.0, -40.0, -43.0], [-13.0, -40.0, -100.0], [-13.0, -40.0, -100.0]
#     ])




# def get_conv_hull_faces(points, verbose=False):
#     hull = ConvexHull(points)
#     simplices = hull.simplices

#     planes = {} # stores the normal vector and a point of the ith simplex

#     for i in range(len(simplices)):

#         p1 = points[simplices[i][0]]
#         p2 = points[simplices[i][1]]
#         p3 = points[simplices[i][2]]

#         normal = np.cross(p2 - p1, p3 - p1)
#         normal = normal / np.linalg.norm(normal) 

#         planes[i] = (normal, p1) 

#     face_groups = {}
#     changed = set() # contains all the changed points

#     for i in range(len(planes)):
#         if planes[i] == -1:
#             continue

#         normal, point = planes[i]
#         face_groups[i] = simplices[i]

#         for j in range(i + 1, len(planes)):
#             if planes[j] == -1:
#                 continue
#             normal2, point2 = planes[j]

#             # print(f"{simplices[i]} and {simplices[j]}")

#             # if np.dot(normal, point - point2) == 0 and abs(np.dot(normal, normal2)) == 1:
#             dot_points = np.dot(normal, point - point2)
#             dot_normals = abs(np.dot(normal, normal2))
#             # print(f"Dot points: {dot_points}, Dot normals: {dot_normals}")
#             if np.isclose(dot_points, 0) and np.isclose(dot_normals, 1):
#                 if verbose:
#                     print(f"Normal 1: {normal}, Point 1: {point}")
#                     print(f"Normal 2: {normal2}, Point 2: {point2}")
#                 face_groups[i] = np.union1d(face_groups[i], simplices[j])
#                 planes[j] = -1
#                 changed = changed.union(set(face_groups[i]))
#             # print()

#     if verbose:
#         print(f"Face groups: {face_groups}")

#     faces = [] # contains the faces with vertices properly ordered

#     for i in face_groups:
#     # centroid approach, each vertex is ordered by angle with respect to the centroid obtained through the mean of the vertices
#         # if len(face_groups[i]) == 3:
#         #     faces.append([int(face_groups[i][0]), int(face_groups[i][1]), int(face_groups[i][2])])
#         #     continue

#         # Get the normal vector for this face
#         normal = planes[i][0]
        
#         # Calculate centroid
#         centroid_x = np.mean(points[face_groups[i], 0])
#         centroid_y = np.mean(points[face_groups[i], 1])
#         centroid_z = np.mean(points[face_groups[i], 2])
#         centroid = np.array([centroid_x, centroid_y, centroid_z])
        
#         # Find the major axis of the normal vector
#         major_axis = np.argmax(np.abs(normal))
#         other_axes = [0, 1, 2]
#         other_axes.remove(major_axis)
        
#         angles_points = []
        
#         for j in range(len(face_groups[i])):
#             # Get the point
#             point = points[face_groups[i][j]]
            
#             # Project the point onto the plane defined by the other two axes
#             projected_x = point[other_axes[0]] - centroid[other_axes[0]]
#             projected_y = point[other_axes[1]] - centroid[other_axes[1]]
            
#             # Calculate the angle in the projected 2D space
#             angle = np.arctan2(projected_y, projected_x)
#             angles_points.append((angle, face_groups[i][j]))
        
#         # Sort angles_points by angle
#         angles_points.sort(key=lambda x: x[0])
        
#         faces.append([int(angles_points[j][1]) for j in range(len(angles_points))])

#     if verbose:
#         print(f"Faces: {faces}")
#     return faces, changed

# Copilot
def get_conv_hull_faces(points, verbose=False):
    hull = ConvexHull(points)
    simplices = hull.simplices

    # Use the hull equations directly (Ax + By + Cz + D = 0 format)
    hull_equations = hull.equations  # Each row is [A, B, C, D]

    # Group faces by their plane equation
    face_groups = {}
    
    # First, normalize all plane equations for consistent comparison
    normalized_equations = []
    for eq in hull_equations:
        # Extract normal vector (A,B,C) and D component
        normal = eq[:3]
        D = eq[3]
        norm = np.linalg.norm(normal)
        
        # Normalize and ensure consistent orientation (first non-zero component positive)
        if norm > 0:
            normal = normal / norm
            D = D / norm
            
            # Ensure consistent sign
            for i in range(3):
                if abs(normal[i]) > 1e-10: # checking if it is not zero
                    if normal[i] < 0: # if the first significant component is negative, flip the signs of everything
                        normal = -normal 
                        D = -D
                    break
                    
        normalized_equations.append((normal, D))
    
    # Group simplices by their normalized plane equations
    for i, (normal, D) in enumerate(normalized_equations):
        # Create a hashable key for the plane equation (with rounding for stability)
        key = tuple(np.round(np.append(normal, D), 9))
        
        if key in face_groups:
            face_groups[key].append(i)
        else:
            face_groups[key] = [i]
    
    # Merge triangles in each group
    merged_faces = []
    changed = set()
    
    for plane_eq, simplex_indices in face_groups.items():
        # Get all vertices from this group of coplanar triangles
        vertices = set()
        for idx in simplex_indices:
            vertices.update(simplices[idx])
            changed.update(simplices[idx])  # Track changed vertices
        
        # Get this face's normal
        normal = np.array(plane_eq[:3])
        
        # Order vertices around perimeter
        vertices_list = list(vertices)
        
        # Project points onto the dominant plane
        major_axis = np.argmax(np.abs(normal))
        other_axes = [i for i in range(3) if i != major_axis]
        
        # Project and sort by angle from centroid
        centroid = np.mean(points[vertices_list], axis=0)
        
        angle_points = []
        for v in vertices_list:
            point = points[v]
            projected = [point[other_axes[0]] - centroid[other_axes[0]],
                        point[other_axes[1]] - centroid[other_axes[1]]]
            angle = np.arctan2(projected[1], projected[0])
            angle_points.append((angle, v))
        
        # Sort by angle
        angle_points.sort()
        
        # Extract ordered vertices
        ordered_face = [p[1] for p in angle_points]
        merged_faces.append(ordered_face)
    
    if verbose:
        print(f"Number of merged faces: {len(merged_faces)}")
    
    return merged_faces, changed


def draw_polytope(points, faces, changed, only_hull_points=False, cut_edges=None, c=None):
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')


    if only_hull_points:
        hull_vertices = set()
        for face in faces:
            hull_vertices.update(face)
        hull_vertices = list(hull_vertices)
        
        # Plot only points on the hull
        ax.scatter(points[hull_vertices, 0], points[hull_vertices, 1], points[hull_vertices, 2], color='r', s=50)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', s=50)

    # plot changed points with a different color
    ax.scatter(points[list(changed), 0], points[list(changed), 1], points[list(changed), 2], color='g', s=1000)

    # Prepare face polygons
    polygons = []
    for idx, face in enumerate(faces):
        # Get the coordinates of each vertex in the face
        polygon = [points[i] for i in face]
        polygons.append(polygon)

    # Create the 3D polygons
    colormap = plt.get_cmap('prism', len(faces))
    face_colors = [colormap(i) for i in range(len(faces))]
    poly = Poly3DCollection(polygons, alpha=0.6, linewidth=4, edgecolor='k')
    poly.set_facecolor(face_colors)

    # Add the collection to the plot
    ax.add_collection3d(poly)

    for idx, face in enumerate(faces):
        polygon = np.array([points[i] for i in face])
        centroid = polygon.mean(axis=0)
        ax.text(*centroid, str(idx), color='black', fontsize=20, ha='center', va='center')

    # Highlight cut edges if provided
    if cut_edges is not None:
        for v1, v2 in cut_edges:    
            # Draw the cut edge with a distinctive color
            ax.plot([points[v1][0], points[v2][0]], [points[v1][1], points[v2][1]], [points[v1][2], points[v2][2]], color='yellow', linewidth=5, zorder=10)

    # Visualize the direction vector c if provided
    if c is not None:
        # Calculate the centroid of the polytope
        centroid = np.mean(points, axis=0)
        
        # Scale the vector for better visibility
        scale = np.max(np.abs(points)) * 0.5
        
        # Draw the direction vector from centroid
        ax.quiver(centroid[0], centroid[1], centroid[2], c[0], c[1], c[2], color='blue', linewidth=3, length=scale, normalize=True, arrow_length_ratio=0.15)
        
        # Add a text label for the vector
        end_point = centroid + scale * c/np.linalg.norm(c)
        ax.text(end_point[0], end_point[1], end_point[2], "c", color='blue', fontsize=15)
        
    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.title('Convex Hull with Merged Faces')
    plt.tight_layout()
    plt.show()

    with open('points.txt', 'w') as f:
        f.write("points = np.array([\n")
        for i in range(len(points)):
            f.write(f"    {points[i].tolist()},\n")
        f.write("])\n")
       
       
if __name__ == "__main__": 
    # points = generate_cube()
    # points = generate_dodec()
    points = generate_uniform(100)
    faces, changed = get_conv_hull_faces(points)
    draw_polytope(points, faces, changed)