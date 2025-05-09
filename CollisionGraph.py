import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# Making graph
NUM_SAMPLES = 20
MAX_GEN = 100
for folder in list(os.walk("results_demo/"))[0][1]:
    # print(folder)
    collisions_avg = np.loadtxt(f"results_demo/{folder}/avg_fitnesses.txt")
    collisions_best = np.loadtxt(f"results_demo/{folder}/best_fitnesses.txt")
    # print(len(collisions))
    # print(len(collisions[0]))
    collisions_overall_best = np.mean(collisions_best, axis=0)
    collisions_overall_avg = np.mean(collisions_avg, axis=0)
    generations = list(range(1, MAX_GEN + 1))
    colors_best = cm.Oranges(np.linspace(0.3, 1, NUM_SAMPLES))
    colors_avg = cm.Blues(np.linspace(0.3, 1, NUM_SAMPLES))
    # colors_avg = cm.Blues(np.linspace(0, 1, NUM_ITERS))

    for k in range(NUM_SAMPLES):
        plt.plot(generations, collisions_best[k], color=colors_best[k], linestyle = "--", alpha=0.5)
        plt.plot(generations, collisions_avg[k], color=colors_avg[k], linestyle = "--", alpha=0.5)

    plt.plot(generations, collisions_overall_best, label="Best Collisions", color='saddlebrown', linewidth=2.5)
    plt.plot(generations, collisions_overall_avg, label="Average Collisions", color='darkblue', linewidth=2.5)

    plt.xlabel('Generation')
    plt.ylabel('Collisions')
    plt.title(f'Best Collisions over Generations - {folder}')
    plt.legend()
    plt.show()