import numpy as np
import sklearn.metrics
from helper_functions import *
from sklearn.model_selection import KFold

filepath_data_100 = 'c:/Users/Kathan/Desktop/AML/Assignment3/datafile/three_class_100.mat'
filepath_labels_100 = 'c:/Users/Kathan/Desktop/AML/Assignment3/datafile/three_label_100.mat'
y = mat_to_array(filepath_labels_100,1,100,'three_label_100')

x = mat_to_array(filepath_data_100,0,100,'three_class_100')
#x_folds = cross_validation_split(x,10)

kf = KFold(n_splits=10)
kf.get_n_splits(x)
x_train = []
X_test = []
y_train = []
y_test = []
for (train_index, test_index), i in kf.split(x),range(10):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train[i], X_test[i] = x[train_index], x[test_index]
    y_train[i], y_test[i] = y[train_index], y[test_index]



#print(np.shape(x_folds))
#print(x_folds[1])

#xfd1 = []
'''
for k in range(10):
    for i in range(9):
        for j in range(10):
            xfd1[k].append(x_folds[i][j])
    
xfl1 = x_folds[9]
'''
#print(np.shape(xfd1))
#print(np.shape(xfl1))


# %%

import matplotlib.pyplot as plt
import h5py
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.datasets.samples_generator import make_blobs
import numpy as np


filepath_data_100 = 'c:/Users/Kathan/Desktop/AML/Assignment3/datafile/three_class_100.mat'
filepath_labels_100 = 'c:/Users/Kathan/Desktop/AML/Assignment3/datafile/three_label_100.mat'
filepath_data_10k = 'c:/Users/Kathan/Desktop/AML/Assignment3/datafile/three_class_10k.mat'
filepath_labels_10K = 'c:/Users/Kathan/Desktop/AML/Assignment3/datafile/three_label_10k.mat'

def mat_to_array(filepath,flag,N,name):
    '''
    Function that converts mat file to numpy array.
    filepath: path to the mat file.
    flag: 0-data |  1-label
    n: number of samples
    name: name of mat file ins string format
    Author: Kathan Vyas
    '''
    if flag == 0:
        x_y_numpy_array = np.zeros((N, 2), dtype=float)
    else:
        x_y_numpy_array = np.zeros((N, 1), dtype=float)

    
    with h5py.File(filepath, 'r') as f:
        for idx, element in enumerate(f[name]):
            x_y_numpy_array[idx] = element[:]
    return x_y_numpy_array   
def cross_validation_split(dataset, folds=3):
    '''
    Function performs k-fold cross validation
    dataset: numpy array
    folds: number of folds
    Author: Kathan Vyas
    '''
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split    



x1 = mat_to_array(filepath_data_100,0,100,'three_class_100')
y1 = mat_to_array(filepath_labels_100,1,100,'three_label_100')
x2 = mat_to_array(filepath_data_10k,0,10000,'three_class_10k')
y2 = mat_to_array(filepath_labels_10K,1,10000,'three_label_10k')

print(np.shape(x1))
print(np.shape(x2))


#%% 

f1 = plt.figure(1)
plt.scatter(x1[:,0],x1[:,1])
f1.show()

g1 = plt.figure(2)
plt.scatter(x2[:,0],x2[:,1])
g1.show()



f2 = plt.figure(3)
plt.scatter(x1[:,0],x1[:,1], c = y1[:,0])
f2.show()

g2 = plt.figure(4)
plt.scatter(x2[:,0],x2[:,1], c = y2[:,0])
g2.show()





# %%
