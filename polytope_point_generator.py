import numpy as np
import random

def generate_cube():
    # a cube
    points = np.array([[0, 0, 0], 
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]])
    return points

import numpy as np

def generate_penta_pyramid(radius=1.0, height=1.0):
    # Base: 5 vertices of a regular pentagon in the XY plane
    angles = np.linspace(0, 2 * np.pi, 6)[:-1]  # 5 points
    base = np.array([[radius * np.cos(a), radius * np.sin(a), 0] for a in angles])

    # Apex: directly above center of base
    apex = np.array([[0, 0, height]])

    # Combine vertices
    points = np.vstack((base, apex))  # Shape: (6, 3)
    return points

def generate_dodec():
    # a dodechahedron
    phi = (1 + np.sqrt(5)) / 2

    points = np.array([

        [ 1,  1,  1],
        [ 1,  1, -1],
        [ 1, -1,  1],
        [ 1, -1, -1],
        [-1,  1,  1],
        [-1,  1, -1],
        [-1, -1,  1],
        [-1, -1, -1],
        
        [0,  1/phi,  phi],
        [0,  1/phi, -phi],
        [0, -1/phi,  phi],
        [0, -1/phi, -phi],
        
        [ 1/phi,  phi, 0],
        [ 1/phi, -phi, 0],
        [-1/phi,  phi, 0],
        [-1/phi, -phi, 0],
        
        [ phi, 0,  1/phi],
        [ phi, 0, -1/phi],
        [-phi, 0,  1/phi],
        [-phi, 0, -1/phi]
    ])
    
    return points


def generate_polytope(COUNT):
    # random.seed(44) # if want to retest polygon
    points = np.empty((COUNT, 3))

    for i in range(COUNT):
        # points[i, 0] = random.uniform(-COUNT, COUNT)
        # points[i, 1] = random.uniform(-COUNT, COUNT)
        # points[i, 2] = random.uniform(-COUNT, COUNT)
        
        if random.uniform(0, 1) < 0.7:
            points[i, 0] = points[i - 1, 0]  
            points[i, 1] = points[i - 1, 1] 
            points[i, 2] = -COUNT
        else:
            points[i, 0] = random.uniform(-COUNT, COUNT)
            points[i, 1] = random.uniform(-COUNT, COUNT)
            points[i, 2] = random.uniform(-COUNT, COUNT)
    return points
