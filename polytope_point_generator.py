import numpy as np
import random
import math

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

def generate_uniform(COUNT):
    # random.seed(44) # if want to retest polygon
    points = np.empty((COUNT, 3))

    for i in range(COUNT):
        # randomly generate points
        points[i, 0] = random.uniform(-COUNT, COUNT)
        points[i, 1] = random.uniform(-COUNT, COUNT)
        points[i, 2] = random.uniform(-COUNT, COUNT)
        
    return points

def generate_turtle(i, j):

    # if i > j:
    #     i, j = j, i

    # i = max(1, i)
    # i = min(7, i)
    # j = max(1, j)
    # j = min(7, j)

    points = []

    for x in range(-i, i + 1):
        for y in range(-j, j + 1):
            points.append([x, y, -1*(x**2 + y**2)]) # making the z coord negative so the turtle is straight up
    
    points = np.array(points)
    return points

def generate_flat(COUNT):

    points = np.empty((COUNT, 3))

    points[0, 0] = random.uniform(-COUNT, COUNT)
    points[0, 1] = random.uniform(-COUNT, COUNT)
    points[0, 2] = random.uniform(-COUNT, COUNT)

    for i in range(1, COUNT):
        # randomly generate points, with a 70% chance of a point being on the same line as the previous
        if random.uniform(0, 1) < 0.7:
            points[i, 0] = points[i - 1, 0]  
            points[i, 1] = points[i - 1, 1] 
            points[i, 2] = -COUNT
        else:
            points[i, 0] = random.uniform(-COUNT, COUNT)
            points[i, 1] = random.uniform(-COUNT, COUNT)
            points[i, 2] = random.uniform(-COUNT, COUNT)
    
    return points

def generate_spherical(COUNT):
    
    points = np.empty((COUNT, 3))

    for i in range(COUNT):
    
        phi = random.uniform(0, np.pi)
        theta = random.uniform(0, 2 * np.pi)

        points[i, 0] = np.sin(phi) * np.cos(theta)
        points[i, 1] = np.sin(phi) * np.sin(theta)
        points[i, 2] = np.cos(phi) 

    return points


def generate_half_spherical(COUNT):
    
    points = np.empty((COUNT, 3))

    for i in range(COUNT):

        phi = random.uniform(0, np.pi/2)
        theta = random.uniform(0, 2 * np.pi)

        points[i, 0] = np.sin(phi) * np.cos(theta)
        points[i, 1] = np.sin(phi) * np.sin(theta)
        points[i, 2] = np.cos(phi) 

    return points