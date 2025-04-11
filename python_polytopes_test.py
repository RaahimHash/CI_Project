from scipy.spatial import ConvexHull
import numpy as np
import random 

def check_if_planar(i, j, k, p):

    A = np.array([i, j, k, p])
    det = np.linalg.det(A)
    
    return det == 0

points = np.empty((10, 3))

for i in range(10):
    points[i, 0] = random.uniform(-10, 10)
    points[i, 1] = random.uniform(-10, 10)
    points[i, 2] = random.uniform(-10, 10)

# # a cube
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
faces = []

for i in range(len(simplices) - 1):
    a = simplices[i]

    for j in range(i + 1, len(simplices)):
        b = simplices[j]
        intersection = np.intersect1d(a, b)

        if len(intersection) != 0: # some point/s overlap

            diff = np.setdiff1d(b, intersection) # get those that don't
            coplanar = True
    
            for k in range(len(diff)): # and check if they are coplanar
                if check_if_planar(points[a[0]], points[a[1]], points[a[2]], points[diff[k]]):
                    coplanar = False
                    break

            if coplanar:
                face = []
                union = np.union1d(a, b) 

                # pick a point, then the next point that is closest to it, and so on
                # this should give us the face in a way that the adjacent indices have an edge (as well as the first and last)
                face.append(union[0]) 
                min_dist = np.inf
                min_i = -1

                while len(face) != len(union):

                    for k in range(len(union)): # check other points to find the closest one
                        if union[k] not in face:
                            dist = np.linalg.norm(points[face[-1]] - points[union[k]])
                            if dist < min_dist:
                                min_dist = dist
                                min_i = union[k]

                    face.append(min_i)
                    min_dist = np.inf
                    min_i = -1

                faces.append(face)

            # need to do multiple passes of this approach, but removing the simplices that are processed and replacing them with the merged face