# Torsion Data Clustering Script

This script processes torsion data, clusters it using DBSCAN, and outputs filtered torsion and conformer data based on the clustering results. The script also normalizes and adjusts torsion values based on symmetry, and extracts energy values from an input XYZ file.

## Usage

```bash
python script.py <torsion_file> <xyz_file> <symmetry_file>

```
<torsion_file>: Input file containing torsion data i.e, torsion values for all the conformers. <br>
<xyz_file>: Input XYZ file containing conformer information. It contains xyz coordinates for all atoms of each conformer, it also contains energy for that conformer. <br>
<symmetry_file>: Input file containing symmetry values for all torsions. for eg. if some torsion have 3-fold symmetry then 3

# Script Steps

## Input Parsing:
The script expects three input files: torsion data, XYZ data, and symmetry data. <br>

## Energy Extraction:
Extracts energy values from the XYZ file.

## Data Processing:
Reads symmetry values and adjusts torsion values based on these values.<br>
Normalizes the torsion data and scales the torsion values based on symmetry.

## DBSCAN Clustering:
Determines the number of clusters for each epsilon value and plots the results.<br>
Prompts the user to select an epsilon value based on the plot.<br>
Applies DBSCAN clustering with the chosen epsilon value.<br>
Adds cluster labels and energy values to the torsion data.<br>
Filters and sorts the data based on energy values.

## Output:
Saves the filtered torsion data to filtered_TORSION and filtered_torsion_data.csv.<br>
Extracts and saves the corresponding conformers from the XYZ file to filtered_temp.xyz.

## Output Files
filtered_TORSION: Contains the filtered torsion data without energy values.<br>
filtered_torsion_data.csv: Contains the filtered torsion data with energy values.<br>
filtered_temp.xyz: Contains the filtered conformers based on the clustering results.

## Example
To run the script, use the following command:

```bash
python process_torsion.py TORSION conformers.xyz symmetry_values.txt
```
# Dependencies
pandas<br>
numpy<br>
scikit-learn<br>
matplotlib

Make sure to install the required dependencies before running the script:
```bash
pip install pandas numpy scikit-learn matplotlib
```
# Notes
Ensure that the input files are formatted correctly and contain valid data.<br>
The script handles ValueError exceptions during energy extraction and checks for matching symmetry values.<br>
Adjust the min_samples parameter for DBSCAN if needed.
