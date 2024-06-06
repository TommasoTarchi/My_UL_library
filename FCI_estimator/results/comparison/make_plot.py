import numpy as np
import os
import json
import matplotlib.pyplot as plt


# Define minimum and maximum intrinsic dimension to plot
min_file_id = 2
max_file_id = 1024


data = {'FCI': {'optimal': []},
        'TwoNN': {'optimal': []}}
ids = []

json_files = [f for f in os.listdir('.') if f.endswith('.json')]
json_files_sorted = sorted(json_files, key=lambda x: int(os.path.splitext(x)[0]))

for filename in json_files_sorted[int(np.log2(min_file_id))-1:]:
    file_id = int(os.path.splitext(filename)[0])  # Extracting ID from filename

    if file_id > max_file_id:
        continue

    ids.append(file_id)

    file_path = os.path.join('.', filename)
    with open(file_path, 'r') as file:
        json_data = json.load(file)
        data['FCI']['optimal'].append(json_data['FCI']['optimal'])
        data['TwoNN']['optimal'].append(json_data['TwoNN']['optimal'])

plt.figure(figsize=(10, 6))

plt.plot(ids, data['FCI']['optimal'], label='FCI', marker='o')
plt.plot(ids, data['TwoNN']['optimal'], label='TwoNN', marker='x')

x_values = np.linspace(min(ids), max(ids), 100)
plt.plot(x_values, x_values, color='red', linestyle='--', linewidth=1, label='expected')

plt.xlabel('True ID')
plt.ylabel('Estimated ID')
plt.legend()
plt.grid(True)
plt.savefig('all_IDs.png')
