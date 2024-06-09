import os
import json
import matplotlib.pyplot as plt


# define minimum and maximum size to plot
min_file_id = 600
max_file_id = 2000


data = {'C': {'optimal': [], 'std_dev': []},
        'B': {'optimal': [], 'std_dev': []}}
ids = []

json_files = [f for f in os.listdir('.') if f.endswith('.json')]
json_files_sorted = sorted(json_files, key=lambda x: int(os.path.splitext(x)[0]))

for filename in json_files_sorted[int(min_file_id/50) - 1:]:
    file_id = int(os.path.splitext(filename)[0])  # Extracting ID from filename

    if file_id > max_file_id:
        continue

    ids.append(file_id)

    file_path = os.path.join('.', filename)
    with open(file_path, 'r') as file:
        json_data = json.load(file)
        data['C']['optimal'].append(json_data['C']['optimal'])
        data['C']['std_dev'].append(json_data['C']['std_dev'])
        data['B']['optimal'].append(json_data['B']['optimal'])
        data['B']['std_dev'].append(json_data['B']['std_dev'])

plt.figure(figsize=(10, 6))
plt.errorbar(ids, data['C']['optimal'], yerr=data['C']['std_dev'], fmt='o', label='Dataset C', capsize=5)
plt.axhline(y=400, color='red', linestyle='--', linewidth=1, label=f'True ID')
plt.xlabel('Dataset size')
plt.ylabel('Estimated ID')
plt.legend()
plt.grid(True)
plt.savefig('C_data.png')

plt.figure(figsize=(10, 6))
plt.errorbar(ids, data['B']['optimal'], yerr=data['B']['std_dev'], fmt='s', label='Dataset B', capsize=5)
plt.axhline(y=15, color='red', linestyle='--', linewidth=1, label='True ID')
plt.xlabel('Dataset size')
plt.ylabel('Estimated ID')
plt.legend()
plt.grid(True)
plt.savefig('B_data.png')
