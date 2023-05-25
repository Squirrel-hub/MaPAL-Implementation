# notebook setup
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import sys
sys.path.append('../../')

# import required packages
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)
#rc('font', family='serif')
rc('axes', edgecolor="black")

from annotlib.cluster_based import ClusterBasedAnnot

from scipy.spatial import ConvexHull

import sys
sys.path.insert(0,'/content/MaPAL-Implementation')

from src.utils.data_functions import investigate_data_set, load_data, preprocess_2d_data_set
from src.utils.plot_functions import plot_simulation

from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# set random seeds
np.random.seed(42)
random_state = np.random.RandomState(40)

# obtain available data sets
data_set_names = pd.read_csv('/content/MaPAL-Implementation/data/data_set_ids.csv').name.values

# simulation for each data set
for d in data_set_names:
    print(d)
    # load data
    X, y_true, y = load_data(d)
    X = X.values
    n_features = X.shape[1]
    n_classes = len(np.unique(y_true))

    # standardize features
    X_trans = StandardScaler().fit_transform(X)


    print('n-features: {}, n-classes: {}'.format(n_features, n_classes))
    
    for s in ['x', 'o', 'y']:
        if s == 'x':
            # simulate annotators with instance-dependent performance values
            n_annotators = np.random.choice([4, 5, 6])
            y_cluster_k_means = KMeans(n_clusters=n_annotators, n_init=50, random_state=6).fit_predict(X_trans)
            
            U = np.random.uniform(0.8, 1.0, size=2*n_annotators).reshape(n_annotators, 2)
            E = np.array([np.arange(n_annotators), 
                          (np.arange(n_annotators)+2)%n_annotators]).reshape((n_annotators, 2))
            A = np.random.uniform(1/n_classes, 1/n_classes+0.2, n_annotators**2)
            A = A.reshape((n_annotators, n_annotators))
            A[np.arange(n_annotators), E[:, 0]] = U[:, 0]
            A[np.arange(n_annotators), E[:, 1]] = U[:, 1]
            C = np.empty((n_annotators, n_annotators, 2))
            C[:, :, 0] = A
            C[:, :, 1] = A
            annot = ClusterBasedAnnot(X=X_trans, y_true=y_true, y_cluster=y_cluster_k_means, n_annotators=n_annotators,
                                      cluster_labelling_acc=C, random_state=6)
        elif s == 'y':
            # simulate annotators with class-dependent performance values
            n_annotators = random_state.choice([4, 5, 6])
            A = random_state.uniform(1/n_classes, 1, size=n_annotators*n_classes).reshape((n_annotators, n_classes))
            C = np.empty((n_annotators, n_classes, 2))
            C[:, :, 0] = A
            C[:, :, 1] = A
            annot = ClusterBasedAnnot(X=X_trans, y_true=y_true, y_cluster=y_true, n_annotators=n_annotators,
                                      cluster_labelling_acc=C, random_state=6)
            
        elif s == 'o':
            # simulate annotators with uniform performance values
            n_annotators = np.random.choice([4, 5, 6])
            y_cluster_const = np.zeros(len(X), dtype=int)
            min_label_acc = 1. / n_classes
            label_acc_step = (0.9 - min_label_acc) / (n_annotators + 1)
            mean_label_acc = np.linspace(min_label_acc, 0.9 - 2 * label_acc_step, n_annotators)
            C = np.empty((n_annotators, 1, 2))
            for a in range(n_annotators):
                v = np.random.uniform(mean_label_acc[a], mean_label_acc[a] + 2 * label_acc_step)
                C[a, :, :] = v
            annot = ClusterBasedAnnot(X=X_trans, y_true=y_true, y_cluster=y_cluster_const, n_annotators=n_annotators,
                                      cluster_labelling_acc=C, random_state=6)
    
        # store data set with simulated annotations
        data_set = {'x_{}'.format(i): X[:, i] for i in range(X.shape[1])}
        data_set['y'] = y_true
        for a in range(annot.n_annotators()): 
            data_set['y_'+str(a+1)] = annot.Y_[:, a]
        data_set = pd.DataFrame(data_set)
        data_set_name = '{}-simulated-{}'.format(d, s)
        filename = '/content/MaPAL-Implementation/data/{}.csv'.format(data_set_name)
        is_file_present = glob.glob(filename)
        if not is_file_present:
            data_set.to_csv(filename, index=False)
        print(s+': '+str(investigate_data_set(data_set_name=data_set_name)))

X, y_true = make_blobs(n_samples=300, n_features=2, centers=12, cluster_std=1, random_state=42)
X -= np.mean(X, keepdims=True, axis=0)
X /= np.std(X, keepdims=True, axis=0)
y_true %= 2
n_classes = len(np.unique(y_true))

np.random.seed(42)
n_annotators = 4
y_cluster_const = np.zeros(len(X), dtype=int)
min_label_acc = 1. / n_classes
label_acc_step = (0.9 - min_label_acc) / (n_annotators + 1)
mean_label_acc = np.linspace(min_label_acc, 0.9 - 2 * label_acc_step, n_annotators)
C = np.empty((n_annotators, 1, 2))
for a in range(n_annotators):
    v = np.random.uniform(mean_label_acc[a], mean_label_acc[a] + 2 * label_acc_step)
    C[a, :, :] = v
print(C)
annot = ClusterBasedAnnot(X=X, y_true=y_true, y_cluster=y_cluster_const, n_annotators=n_annotators,
                          cluster_labelling_acc=C, random_state=0)

# save data set
data_set = {'x_{}'.format(i): X[:, i] for i in range(X.shape[1])}
data_set['y'] = y_true
for a in range(annot.n_annotators()): 
    data_set['y_'+str(a+1)] = annot.Y_[:, a]
data_set = pd.DataFrame(data_set)
data_set_name = 'example-data-set-o'
filename = '../../data/{}.csv'.format(data_set_name)
data_set.to_csv(filename, index=False)
fig = plot_simulation(X=X, y_true=y_true, y=annot.Y_, figsize=(24, 12), fontsize=30,
                      filename='../../plots/simulated-o', filetype='svg')

np.random.seed(42)
n_annotators = 4
y_cluster_const = np.zeros(len(X), dtype=int)
A = np.random.uniform(1/n_classes, 1, size=n_annotators*n_classes).reshape((n_annotators, n_classes))
C = np.empty((n_annotators, n_classes, 2))
C[:, :, 0] = A
C[:, :, 1] = A
annot = ClusterBasedAnnot(X=X, y_true=y_true, y_cluster=y_true, n_annotators=n_annotators,
                          cluster_labelling_acc=C, random_state=0)
print(C[:, :, 0])

# save data set
data_set = {'x_{}'.format(i): X[:, i] for i in range(X.shape[1])}
data_set['y'] = y_true
for a in range(annot.n_annotators()): 
    data_set['y_'+str(a+1)] = annot.Y_[:, a]
data_set = pd.DataFrame(data_set)
data_set_name = 'example-data-set-y'
filename = '../../data/{}.csv'.format(data_set_name)
data_set.to_csv(filename, index=False)
fig = plot_simulation(X=X, y_true=y_true, y=annot.Y_, figsize=(24, 12), fontsize=30,
                      filename='../../plots/simulated-y', filetype='svg')

np.random.seed(42)
n_annotators = 4
y_cluster_k_means = KMeans(n_clusters=n_annotators, n_init=50, random_state=0).fit_predict(X)
print(np.unique(y_cluster_k_means, return_counts=True))

U = np.random.uniform(0.8, 1.0, size=2*n_annotators).reshape(n_annotators, 2)
E = np.array([np.arange(n_annotators), 
              (np.arange(n_annotators)+2)%n_annotators]).reshape((n_annotators, 2))
A = np.random.uniform(1/n_classes, 1/n_classes+0.2, n_annotators**2)
A = A.reshape((n_annotators, n_annotators))
A[np.arange(n_annotators), E[:, 0]] = U[:, 0]
A[np.arange(n_annotators), E[:, 1]] = U[:, 1]
C = np.empty((n_annotators, n_annotators, 2))
C[:, :, 0] = A
C[:, :, 1] = A
print(C[:, :, 0])
print(np.unique(y_cluster_k_means, return_counts=True))

# simulate annotators
annot = ClusterBasedAnnot(X=X, y_true=y_true, y_cluster=y_cluster_k_means, n_annotators=n_annotators, 
                          cluster_labelling_acc=C, random_state=0)

# save data set
data_set = {'x_{}'.format(i): X[:, i] for i in range(X.shape[1])}
data_set['y'] = y_true
for a in range(annot.n_annotators()): 
    data_set['y_'+str(a+1)] = annot.Y_[:, a]
data_set = pd.DataFrame(data_set)
data_set_name = 'example-data-set-x'
filename = '../../data/{}.csv'.format(data_set_name)
data_set.to_csv(filename, index=False)
fig = plot_simulation(X=X, y_true=y_true, y=annot.Y_, figsize=(24, 12), fontsize=30,
                      filename='../../plots/simulated-x', filetype='svg', y_cluster=y_cluster_k_means)

X, y_true = make_blobs(n_samples=1000, n_features=2, centers=12, cluster_std=1, random_state=42)
X -= np.mean(X, keepdims=True, axis=0)
X /= np.std(X, keepdims=True, axis=0)
y_true %= 2
n_classes = len(np.unique(y_true))

np.random.seed(42)
n_annotators = 4
y_cluster_k_means = KMeans(n_clusters=n_annotators, n_init=50, random_state=0).fit_predict(X)
print(np.unique(y_cluster_k_means, return_counts=True))

U = np.random.uniform(0.8, 1.0, size=2*n_annotators).reshape(n_annotators, 2)
E = np.array([np.arange(n_annotators), 
              (np.arange(n_annotators)+2)%n_annotators]).reshape((n_annotators, 2))
A = np.random.uniform(1/n_classes, 1/n_classes+0.2, n_annotators**2)
A = A.reshape((n_annotators, n_annotators))
A[np.arange(n_annotators), E[:, 0]] = U[:, 0]
A[np.arange(n_annotators), E[:, 1]] = U[:, 1]
C = np.empty((n_annotators, n_annotators, 2))
C[:, :, 0] = A
C[:, :, 1] = A
print(C[:, :, 0])
print(np.unique(y_cluster_k_means, return_counts=True))

# simulate annotators
annot = ClusterBasedAnnot(X=X, y_true=y_true, y_cluster=y_cluster_k_means, n_annotators=n_annotators, 
                          cluster_labelling_acc=C, random_state=0)

# save data set
data_set = {'x_{}'.format(i): X[:, i] for i in range(X.shape[1])}
data_set['y'] = y_true
for a in range(annot.n_annotators()): 
    data_set['y_'+str(a+1)] = annot.Y_[:, a]
data_set = pd.DataFrame(data_set)
data_set_name = 'large-example-data-set-x'
filename = '../../data/{}.csv'.format(data_set_name)
data_set.to_csv(filename, index=False)
fig = plot_simulation(X=X, y_true=y_true, y=annot.Y_, figsize=(24, 12), fontsize=30,
                      filename='../../plots/simulated-x-large', filetype='svg', y_cluster=y_cluster_k_means)

