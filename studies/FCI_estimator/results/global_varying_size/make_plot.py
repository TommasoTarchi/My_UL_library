import os
import json
import matplotlib.pyplot as plt


# define minimum and maximum size to plot
min_file_id = 50
max_file_id = 1000


data = {'D': {'optimal': [], 'std_dev': []},
        'G': {'optimal': [], 'std_dev': []},
        'H': {'optimal': [], 'std_dev': []}}
ids = []

json_files = [f for f in os.listdir('.') if f.endswith('.json')]
json_files_sorted = sorted(json_files, key=lambda x: int(os.path.splitext(x)[0]))

for filename in json_files_sorted[int(min_file_id/50) - 1:]:
    file_id = int(os.path.splitext(filename)[0])

    if file_id > max_file_id:
        continue

    ids.append(file_id)

    file_path = os.path.join('.', filename)
    with open(file_path, 'r') as file:
        json_data = json.load(file)
        data['D']['optimal'].append(json_data['D']['optimal'])
        data['D']['std_dev'].append(json_data['D']['std_dev'])
        data['G']['optimal'].append(json_data['G']['optimal'])
        data['G']['std_dev'].append(json_data['G']['std_dev'])
        data['H']['optimal'].append(json_data['H']['optimal'])
        data['H']['std_dev'].append(json_data['H']['std_dev'])

plt.figure(figsize=(10, 6))

plt.errorbar(ids, data['D']['optimal'], yerr=data['D']['std_dev'], fmt='o', label='Dataset D', capsize=5)
plt.errorbar(ids, data['G']['optimal'], yerr=data['G']['std_dev'], fmt='s', label='Dataset G', capsize=5)
plt.errorbar(ids, data['H']['optimal'], yerr=data['H']['std_dev'], fmt='^', label='Dataset H', capsize=5)

plt.axhline(y=400, color='red', linestyle='--', linewidth=1, label='True ID')

plt.xlabel('Dataset size')
plt.ylabel('Estimated ID')
plt.legend()
plt.grid(True)
plt.savefig('all_sizes.png')
