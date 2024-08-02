import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import csv
import sys

# Check if the required number of arguments are provided
if len(sys.argv) < 4:
    print("Usage: python script.py <Torsion file name> <xyz file name> <symmetry file name>")
    sys.exit(1)

# Input file names from command line arguments
input_filename = sys.argv[1]
output_filename = 'torsion_data.csv'
xyz_file_name = sys.argv[2]
symmetry_file_name = sys.argv[3]

# Function to extract energies from xyz files
def extract_energies(file_path):
    energies = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines_per_conformer = int(lines[0].strip()) + 2
    num_conformers = len(lines) // lines_per_conformer

    for i in range(num_conformers):
        start_line = i * lines_per_conformer
        energy_line = start_line + 1
        energy_line_content = lines[energy_line].strip()

        energy_start_index = energy_line_content.find('E=') + 2
        energy_end_index = energy_line_content.find(',', energy_start_index)
        energy_str = energy_line_content[energy_start_index:energy_end_index]

        try:
            energy = float(energy_str)
            energies.append(energy)
        except ValueError:
            print(f"Error extracting energy from Conformer {i+1}")

    return energies

# Extract energies from the xyz file
energies = extract_energies(xyz_file_name)

# Read torsion data from input file and save it to a CSV
data = []
with open(input_filename, 'r') as file:
    for idx, line in enumerate(file, start=1):
        intensities = line.strip().split()
        row = intensities
        data.append(row)

with open(output_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['Serial_Number'] + [f'Torsion_{i}' for i in range(1, 20)]
    writer.writerow(header)
    writer.writerows(data)

# Load torsion data into a DataFrame
data = pd.read_csv("torsion_data.csv")
data_copy = pd.read_csv("torsion_data.csv")
serial_number = data['Serial_Number']
data = data.drop(columns=['Serial_Number'])

# Read symmetry values from the symmetry file
with open(symmetry_file_name, 'r') as file:
    symmetry_values = list(map(int, file.readline().strip().split()))

# Check if the number of symmetry values matches the number of torsion columns
if len(symmetry_values) != len(data.columns):
    print("The number of symmetry values must match the number of torsion columns.")
    sys.exit(1)

# Adjust torsion values based on symmetry
for i, x in enumerate(symmetry_values, start=1):
    data[f'Torsion_{i}'] = (((data[f'Torsion_{i}'].astype(float) - (np.pi / x)) % (2 * (np.pi) / x)) - (np.pi / x))

# Normalize the data
data = (data - data.min()) / (data.max() - data.min())

# Scale torsion values based on symmetry
for i, x in enumerate(symmetry_values, start=1):
    data[f'Torsion_{i}'] = (data[f'Torsion_{i}'].astype(float) * 1/x)

# Define epsilon values for DBSCAN
epsilon_values = np.linspace(0.01, 1, 100)
num_clusters = []

# Determine the number of clusters for each epsilon value
for epsilon in epsilon_values:
    dbscan = DBSCAN(eps=epsilon, min_samples=1)
    clusters = dbscan.fit_predict(data)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    num_clusters.append(n_clusters)

# Plot epsilon vs. number of clusters
plt.plot(epsilon_values, num_clusters, marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Number of Clusters')
plt.title('Epsilon vs. Number of Clusters')
plt.grid(True)
plt.show()

# Get user input for epsilon value based on the plot
epsilon = float(input("Based on epsilon vs clusters graph, decide epsilon: "))

# Set minimum samples for DBSCAN
min_samples = 1

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
clusters = dbscan.fit_predict(data)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)
print("Number of clusters:", n_clusters)

# Add cluster labels to the data
data['Cluster'] = clusters

# Reinsert serial number and add energy values to the data
data.insert(0, 'Serial_Number', serial_number)
data['Energy'] = energies
data_copy['Energy'] = energies

# Get unique clusters and sort by energy
df_unique_clusters = data.groupby('Cluster').first().reset_index()
df_unique_clusters = df_unique_clusters.drop(columns=['Cluster'])
serial_numbers_unique = df_unique_clusters['Serial_Number'].tolist()
filtered_data = data_copy[data_copy['Serial_Number'].isin(serial_numbers_unique)]
filtered_data_sorted = filtered_data.sort_values(by='Energy')
final_data = filtered_data_sorted.drop(columns=['Energy'])

# Save filtered torsion data to files
final_data.to_csv('filtered_TORSION', sep=' ', index=False, header=False)
filtered_data_sorted.to_csv('filtered_torsion_data.csv', index=False)
print("Filtered data saved to filtered_TORSION")

# Load the filtered torsion data
data = pd.read_csv('filtered_torsion_data.csv', header=None)

# Extract the first column and convert it to a list of indexes to keep
first_column = data.iloc[1:, 0]  # Skip the first row if it's a header
first_column_list = first_column.tolist()
indexes_to_keep = [int(float(line.strip()) / 1000) for line in first_column_list]

# Read the xyz file
with open(xyz_file_name, 'r') as xyz_file:
    xyz_lines = xyz_file.readlines()

# Calculate the number of lines per conformer
lines_per_conformer = int(xyz_lines[0].strip()) + 2  # Number of atoms + 2 lines for header and comment
lines_to_keep = []

# Collect corresponding conformers based on indexes to keep
for index in indexes_to_keep:
    start_line = index * lines_per_conformer
    end_line = start_line + lines_per_conformer
    lines_to_keep.extend(xyz_lines[start_line:end_line])

# Write the filtered conformers to a new file
with open('filtered_temp.xyz', 'w') as filtered_file:
    filtered_file.writelines(lines_to_keep)

print("Filtered file 'filtered_temp.xyz' created successfully.")
