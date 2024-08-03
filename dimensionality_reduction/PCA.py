import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

import sys
import time
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        label_sample = self.labels[idx]
        return data_sample, label_sample


def extract_x(file_path):
    data = pd.read_csv(file_path, sep=' ',header=None)
    second_column = data.iloc[:, 1]
    return second_column


def extract_y(file_path):
    data = pd.read_csv(file_path, sep=' ',header=None)
    first_column = data.iloc[:,0]
    return first_column

###begin actual code###
# base_path = sys.argv[1]
# no_struc = int(sys.argv[2])
# test_struc = int(sys.argv[3])

no_struc = int(sys.argv[1])
test_struc = int(sys.argv[2])

#torsion number for which se use the neural network

start_time = time.time()
###Code for extracting all x data###
# file_path = base_path[:-1] + 'big_file_x' + str(no_struc)
file_path ='big_file_x' + str(no_struc)

data = pd.read_csv(file_path, sep='\t',header=None)
x_total = torch.tensor(data.values, dtype=torch.float32)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Extracting x took {elapsed_time:.2f} seconds to execute")

start_time = time.time()
###Code for extracting all y data###
# file_path = base_path[:-1] + 'big_file_y' + str(no_struc)
file_path = 'big_file_y' + str(no_struc)

data2 = pd.read_csv(file_path, sep='\t',header=None)
y_total = torch.tensor(data2.values, dtype=torch.float32)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Extracting y took {elapsed_time:.2f} seconds to execute")

###Binning x data###
#bit weird behaviour, learning is now way slower...
bin_size=10
no_bins=x_total.shape[1]//bin_size
x_total_binned=torch.empty(x_total.shape[0], no_bins)

for i in range(no_bins):
    start = i * bin_size
    end = (i + 1) * bin_size
    chunk = x_total[:, start:end]
    x_total_binned[:, i] = torch.sum(chunk, dim=1)  # Calculate the mean of each bin


start_time = time.time()
max_values, _ = torch.max(abs(x_total_binned), dim=1) #max values for all rows
threshold_values = 0.05 * max_values.unsqueeze(1) #determine threshold max_values
x_total_binned = torch.where(abs(x_total_binned) < threshold_values, torch.tensor(0.0), x_total_binned)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"setting to 0 took {elapsed_time:.2f} seconds to execute")

#now check if there are columns which are just zeroes
start_time = time.time()
non_zero_columns_mask = (x_total_binned != 0).any(dim=0) #a mask that is only false when the whole column contains 0's
x_total_binned = x_total_binned[:,non_zero_columns_mask] #only when the mask is true the column remains, thereby removing all columns with just zeroes
end_time = time.time()
elapsed_time = end_time - start_time
print(f"deleting columns with just 0 took {elapsed_time:.2f} seconds to execute")

###Data duplication, implying symmetry of x,y -> -x,-y
temp_x=torch.zeros(no_struc*no_struc, x_total_binned.shape[1], dtype=x_total_binned.dtype)
temp_y=torch.zeros(no_struc*no_struc, y_total.shape[1], dtype=y_total.dtype)

#Code to get indices
lower_indices_new=[]
lower_indices_old=[]
upper_indices_new=[]
upper_indices_old=[]

for i in range(no_struc):  # Loop over all data points
    for j in range(no_struc):
        if i == j:  # Diagonal, so x and y are 0, which is already the case
            continue
        index_new=i*no_struc+j
        if j > i:  # Upper triangle part, which is negative of part in lower triangle
            index_old=((j*(j-1))//2)+i
            lower_indices_old.append(index_old)
            lower_indices_new.append(index_new)
        if i > j:  # Lower triangle part
            index_old=((i*(i-1))//2)+j
            upper_indices_old.append(index_old)
            upper_indices_new.append(index_new)

#filling of temp tensors
temp_x[lower_indices_new]=-x_total_binned[lower_indices_old]
temp_y[lower_indices_new]=-y_total[lower_indices_old]

temp_x[upper_indices_new]=x_total_binned[upper_indices_old]
temp_y[upper_indices_new]=y_total[upper_indices_old]

#putting data in original parameter again
x_total_binned.resize_(temp_x.shape)
x_total_binned.copy_(temp_x)

y_total.resize_(temp_y.shape)
y_total.copy_(temp_y)

###Data splitting ###
train_indices = []
test_indices = []
for i in range(0,no_struc**2):
    j=i//no_struc
    k=i%no_struc
    if j==test_struc or k == test_struc :
        test_indices.append(i)
    else:
        train_indices.append(i)


x_train = x_total_binned[train_indices]
y_train = y_total[train_indices]

x_test = x_total_binned[test_indices]
y_test = y_total[test_indices]

# Perform pca using sklearn
x_train_numpy=x_train.numpy()
x_test_numpy=x_test.numpy()

#PCA parameters
pca = PCA(n_components=0.95,svd_solver='full')

#normalise data
scaler = StandardScaler(with_std=True)
scaler.fit(x_train_numpy)
x_train_numpy_std = scaler.transform(x_train_numpy)

scaler.fit(x_test_numpy)
x_test_numpy_std = scaler.transform(x_test_numpy)


start_time = time.time()
x_train_pca = pca.fit_transform(x_train_numpy_std)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"PCA calculation took {elapsed_time:.2f} seconds to execute")

x_test_pca = pca.transform(x_test_numpy_std)

num_components = pca.n_components_
print(f"Number of PC components: {num_components}")

x_train_pca = torch.Tensor(x_train_pca)
x_test_pca = torch.Tensor(x_test_pca)

# Get the eigenvaluesa
eigenvalues = pca.explained_variance_ratio_

# Print the eigenvalues
for i, eigenvalue in enumerate(eigenvalues, 1):
    print(f"Eigenvalue {i}: {eigenvalue:.4f}")

#Now all the x data is in x_train_pca and x_test_pca. Each row is a different data point and the columns are the different parameters (spectral ranges)
#All the y data is in y_train and y_test. Each row is a different data point and the columns are different outputs (torsions)
#The test data consists of all points with 1 specific structure. This will be most often the lowest energy structure so S1.
#The train data cnsists of all other points, so there is no data point that corresponds to 1 specific structure S1.

#The only thing that you could change from above is the way how PCA is performed, by changing the n_components to a different percentage or by including or excluding standard deviation scaling.

#Below here you can use any ML algorithm to correlate X and Y. 
#If it is a regression algorithm print the RMSE of the training and testing set and also other statistical measures if you want
#Maybe in the future if you want to also try classification algorithms you should print the confusion matrix and any other statistical measure you want
x_train_df = pd.DataFrame(x_train_pca.numpy())
x_test_df = pd.DataFrame(x_test_pca.numpy())
y_train_df = pd.DataFrame(y_train.numpy())
y_test_df = pd.DataFrame(y_test.numpy())

# # # Save DataFrames to CSV files
x_train_df.to_csv('x_train_pca.csv', index=False)
x_test_df.to_csv('x_test_pca.csv', index=False)
y_train_df.to_csv('y_train.csv', index=False)
y_test_df.to_csv('y_test.csv', index=False)

# print(f"x_train_pca shape: {x_train_pca.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"x_test_pca shape: {x_test_pca.shape}")
# print(f"y_test shape: {y_test.shape}")

