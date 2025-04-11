from scipy.spatial import ConvexHull
import numpy as np
import random 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# def check_if_planar(i, j, k, p):

#     A = np.array([i, j, k, p])
#     det = np.linalg.det(A)
    
#     return det == 0

# points = np.empty((10, 3))

# for i in range(10):
#     points[i, 0] = random.uniform(-10, 10)
#     points[i, 1] = random.uniform(-10, 10)
#     points[i, 2] = random.uniform(-10, 10)

# # # a cube
# # points = np.array([[0, 0, 0], 
# #                   [0, 0, 1],
# #                   [0, 1, 0],
# #                   [0, 1, 1],
# #                   [1, 0, 0],
# #                   [1, 0, 1],
# #                   [1, 1, 0],
# #                   [1, 1, 1]])

# hull = ConvexHull(points)
# simplices = hull.simplices 
# faces = []

# for i in range(len(simplices) - 1):
#     a = simplices[i]

#     for j in range(i + 1, len(simplices)):
#         b = simplices[j]
#         intersection = np.intersect1d(a, b)

#         if len(intersection) != 0: # some point/s overlap

#             diff = np.setdiff1d(b, intersection) # get those that don't
#             coplanar = True
    
#             for k in range(len(diff)): # and check if they are coplanar
#                 if check_if_planar(points[a[0]], points[a[1]], points[a[2]], points[diff[k]]):
#                     coplanar = False
#                     break

#             if coplanar:
#                 face = []
#                 union = np.union1d(a, b) 

#                 # pick a point, then the next point that is closest to it, and so on
#                 # this should give us the face in a way that the adjacent indices have an edge (as well as the first and last)
#                 face.append(union[0]) 
#                 min_dist = np.inf
#                 min_i = -1

#                 while len(face) != len(union):

#                     for k in range(len(union)): # check other points to find the closest one
#                         if union[k] not in face:
#                             dist = np.linalg.norm(points[face[-1]] - points[union[k]])
#                             if dist < min_dist:
#                                 min_dist = dist
#                                 min_i = union[k]

#                     face.append(min_i)
#                     min_dist = np.inf
#                     min_i = -1

#                 faces.append(face)

#             # need to do multiple passes of this approach, but removing the simplices that are processed and replacing them with the merged face


points = np.empty((10, 3))

for i in range(10):
    points[i, 0] = random.uniform(-10, 10)
    points[i, 1] = random.uniform(-10, 10)
    points[i, 2] = random.uniform(-10, 10)

# a cube
# points = np.array([[0, 0, 0], 
#                   [0, 0, 1],
#                   [0, 1, 0],
#                   [0, 1, 1],
#                   [1, 0, 0],
#                   [1, 0, 1],
#                   [1, 1, 0],
#                   [1, 1, 1]])

hull = ConvexHull(points)
simplices = hull.simplices

planes = {} # stores the normal vector and a point of the ith simplex

for i in range(len(simplices)):

    p1 = points[simplices[i][0]]
    p2 = points[simplices[i][1]]
    p3 = points[simplices[i][2]]

    normal = np.cross(p2 - p1, p3 - p1)
    normal = normal / np.linalg.norm(normal) 

    planes[i] = (normal, p1) 

face_groups = {}

for i in range(len(planes)):
    if planes[i] == -1:
        continue

    normal, point = planes[i]
    face_groups[i] = simplices[i]

    for j in range(i + 1, len(planes)):
        normal2, point2 = planes[j]

        # print(f"{simplices[i]} and {simplices[j]}")

        if np.dot(normal, point - point2) == 0 and abs(np.dot(normal, normal2)) == 1:
            # print(f"Normal 1: {normal}, Point 1: {point}")
            # print(f"Normal 2: {normal2}, Point 2: {point2}")
            face_groups[i] = np.union1d(face_groups[i], simplices[j])
            planes[j] = -1
        # print()

print(f"Face groups: {face_groups}")

faces = []

for i in face_groups:
    face = []
    face.append(int(face_groups[i][0]))
    min_dist = np.inf
    min_i = -1

    while len(face) != len(face_groups[i]):
        
        for k in range(len(face_groups[i])):
            if face_groups[i][k] not in face:
                dist = np.linalg.norm(points[face[-1]] - points[face_groups[i][k]])
                if dist < min_dist:
                    min_dist = dist
                    min_i = face_groups[i][k]

        face.append(int(min_i))
        min_dist = np.inf
        min_i = -1
    
    faces.append(face)

print(f"Faces: {faces}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', s=50)

# Prepare face polygons
polygons = []
for face in faces:
    # Get the coordinates of each vertex in the face
    polygon = [points[i] for i in face]
    polygons.append(polygon)

# Create the 3D polygons
poly = Poly3DCollection(polygons, alpha=0.7, linewidth=1, edgecolor='k')
poly.set_facecolor('cyan')

# Add the collection to the plot
ax.add_collection3d(poly)

# Set labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

plt.title('Convex Hull with Merged Faces')
plt.tight_layout()
plt.show()