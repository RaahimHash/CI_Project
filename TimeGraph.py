import os
import json
import matplotlib.pyplot as plt
import numpy as np

base_directory = "results_demo/"

all_num_faces = []
all_time_taken = []
all_folder_indices = []

folders = list(os.walk(base_directory))[0][1]
folder_to_index = {folder: idx for idx, folder in enumerate(folders)}

# Read all JSON files and track the folder origin
for folder in folders:
    folder_index = folder_to_index[folder]
    directory = os.path.join(base_directory, folder)
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            print(filepath)
            with open(filepath, 'r') as file:
                data = json.load(file)
                num_faces = data.get("num_faces", [])
                time_taken = data.get("time_taken", [])
                for i in num_faces:
                    all_num_faces.append(num_faces[i])
                    all_time_taken.append(time_taken[i])
                    all_folder_indices.append(folder_index)

# Convert to numpy arrays
x = np.array(all_num_faces)
y = np.array(all_time_taken)
colors = np.array(all_folder_indices)

# Create scatter plot with colormap
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, y, c=colors, cmap='viridis', label='Data points')

# Line of best fit (linear regression)
# coeffs = np.polyfit(x, y, 1)
# best_fit = np.poly1d(coeffs)
# plt.plot(x, best_fit(x), color='red', label=f'Best fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')

# Labels and legend
plt.xlabel("Number of Faces")
plt.ylabel("Time Taken")
plt.title("Time Taken vs. Number of Faces")

# Create a colorbar with folder names
cbar = plt.colorbar(scatter, ticks=range(len(folders)))
cbar.ax.set_yticklabels(folders)
cbar.set_label('Type of Polygon')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
