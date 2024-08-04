# Files

This repository contains three folders :

## report_mitacs.pdf
This is report of project containing sections INTRODUCTION, DATASETS, PREPROCESSING STEPS, METHODOLOGY AND RESULTS

## preprocess_torsion
This folder contains file process_torsion.py which is used to identify and remove duplicates from TORSION data given by CREST.<br>
Its output will be filtered torsion file and filtered xyz file.<br>
Read /preprocess_torsion/README.md file inside this folder for more details.

## dimensionality_reduction
This folder contains file PCA.py which is applied on big_file_x and big_file_y.<br>
big_file_x contains all the spectral differences among conformers leads to n^2 rows and 4001 columns (representing spectral differences of 0 to 4000 wavenumbers).<br>
big_file_y contains all the torsion differences among conformers leads to n^2 rows and 19 columns (19 torsions).
Output of PCA.py is train_x_pca.csv, train_y_pca.csv, test_x_pca.csv and test_y_pca.csv.

## regression
This folder contains files for training and testing different models (decision tree regressor and random forest regressor) on test and train data genereated by dimensionality_reduction/PCA.py.
It also contain file for hyperparameter tuning of models.
This gives results in Mean Squarred Error.
