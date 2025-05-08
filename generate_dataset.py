from polytope_point_generator import * 

import random
import os 
import numpy as np

categories = {"unif5k": (generate_uniform, (5000,)), "unif2k": (generate_uniform, (2000,)), "flat2k": (generate_flat, (2000,)), "turt": (generate_turtle,), "spher": (generate_spherical, (100,)), "half_spher": (generate_half_spherical, (100,))} 

for category in categories:
    os.makedirs(f"temp_dataset/{category}", exist_ok=True)
    if category != "turt":
        for i in range(3):
            points = categories[category][0](*categories[category][1])
            with open(f"temp_dataset/{category}/{i}.txt", "w") as f:
                np.savetxt(f, points, fmt='%.10f')

    else:
        for i in range(3):
            points = categories[category][0](random.randint(4, 8), random.randint(4, 8))
            with open(f"temp_dataset/{category}/{i}.txt", "w") as f:
                np.savetxt(f, points, fmt='%.10f')