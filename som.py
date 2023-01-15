# -*- coding: utf-8 -*-
"""

A simple implementation of self-organizing maps.

To do:
    - try other data sets

@author: Glenn Bruns
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

def eucl_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


def learning_rate(s, initial=0.1, k=100):
    """ Lower the learning rate by factor of 10 every k steps. """
    
    return initial * (.1**(s/k))


def neighbor_fun(x1, x2, s, c1=0.1, c2=0.5):
    """ Return a value between 0 and 1 reflecting the degree
    to which vectors x1 and x2 are neighbors. This function has
    the following properties:
        - 0 <= neighbor_fun(x1, x2, s) <= 1
        - neighbor_fun(x, x, s) = 1
        - neighbor_fun(x1, x2, s) >= neighbor_fun(x1, x2, s+1)
    """
    
    # compute L1 (manhattan) distance between points
    d = np.linalg.norm(np.array(x1) - np.array(x2), ord=1)
    return np.exp(-(d**2)*c1*((s+1)**c2))
    

class SOM:
    
    """ A simple implementation of self-organizing maps.
    A 2-dimensional output space is assumed. 
    """
    
    def __init__(self, b=4, c=4, nsteps=1000, k=3, 
                 initial_learning_rate=0.25, learning_rate_decay=100, random_state=None):
        self.b = b
        self.c = c
        self.nsteps = nsteps
        self.k = k
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.random_state = random_state
                    
    def fit(self, X):
        """ Compute neuron weights. """
        
        m,n = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # neuron weights
        self.W = np.zeros((self.b, self.c, X.shape[1]))
        # each neuron is initialized to the mean of several random inputs
        for i in range(self.b):
            for j in range(self.c):
                idx = np.random.choice(X.shape[0], size=self.k)
                self.W[i,j] = X[idx].mean(axis=0)
        
        for s in range(self.nsteps):
            # select a random input
            x = X[np.random.choice(m)]
            
            # find the index of the best matching unit (BMU)
            distances = np.linalg.norm(x - som.W, axis=-1)
            bmu = np.unravel_index(distances.argmin(), distances.shape)
            
            # update node weights
            rate = learning_rate(s, initial=self.initial_learning_rate, k=self.learning_rate_decay)
            for i in range(self.b):
                for j in range(self.c):
                    delta = neighbor_fun((i,j), bmu, 2) * rate * (x - self.W[i,j])
                    self.W[i,j] += delta      
                    
    def transform(self, X):
        """ Map each row of X to its 2D representation. """
        
        num_output_dims = 2
        result = np.zeros((X.shape[0], num_output_dims))
    
        # for every input x, find the index of the BMU
        for m in range(X.shape[0]):
            x = X[m]
            distances = np.linalg.norm(x - som.W, axis=-1)
            bmu = np.unravel_index(distances.argmin(), distances.shape)                        
            result[m] = bmu
            
        return result
    
    def quantization_error(self, X):
        """ Return the average error between an input in X and
        the value of the node that X is mapped to. """
        
        bmus = self.transform(X).astype(int)
        # get the values of the BMUs
        bmu_vals = som.W[bmus[:,0], bmus[:,1]]
        # compute Eucl distance between each point and the value of its BMU
        dists = np.linalg.norm(X - bmu_vals, axis=0)
        # the quantization error is the average distance
        quant_err = np.mean(dists)

        return quant_err
     

# generic plotting of output
def plot_grid(bmus, hue=None):

    bins = [ int(bmus[:,i].max() - bmus[:,i].min() + 1) for i in [0,1] ]
    plt.figure(figsize=(3,3))
    df1 = pd.DataFrame(bmus, columns = ['x1', 'x2'])
    if hue is None:
        sns.displot(df1, x='x1', y='x2', bins=bins); 
    else:
        sns.displot(df1, x='x1', y='x2', hue=hue, bins=bins)
        
    xwidth, ywidth = [ (nbins-1)/nbins for nbins in bins ]
    plt.xticks([ xwidth * (i+0.5) for i in range(bins[0]) ], labels=range(bins[0]))
    plt.yticks([ xwidth * (i+0.5) for i in range(bins[1]) ], labels=range(bins[1]))
    
    plt.title('Frequency plot');
        
# =============================================================================
#        
# tests and examples
#
# =============================================================================

# visualize neighborhood function
distances = np.arange(0, 4, 0.1)
for s in [0,10,100,500,1000]:
    ys = []
    for d in distances:
        x1 = np.array([0, 0])
        x2 = np.array([0, d])
        ys.append(neighbor_fun(x1, x2, s, c1=0.1, c2=0.5))
    plt.plot(distances, ys, label=str(s))
plt.title('Neighborhood function value by distances')
plt.xlabel('distance')
plt.ylabel('neighborhood value')
plt.legend(title='number of steps');

# =============================================================================
# Iris example
# =============================================================================

# plot irises in lower-dimensional space
def plot_iris(bmus, species):

    species_icon = {'versicolor': 'ro', 'virginica': 'bs', 'setosa': 'g^'}
    plt.figure(figsize=(3,3))
    plt.title('Irises in reduced space')
    for i in range(bmus.shape[0]):
        bmu = bmus[i]
        plt.plot(bmu[0], bmu[1], species_icon[species[i]], alpha=0.2)

# load the iris data
df = sns.load_dataset('iris')
numeric_vars = df.columns[:4]
X = df.loc[:,numeric_vars].values

# scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# test 1
som = SOM()
som.fit(X)
bmus = som.transform(X)
plot_iris(bmus, df['species'])

# test 2
som = SOM(b=6, c=6)
som.fit(X)
bmus = som.transform(X)
plot_iris(bmus, df['species'])

# test 3
som = SOM(b=6, c=6, nsteps=2000)
som.fit(X)
bmus = som.transform(X)
plot_iris(bmus, df['species'])

# test 4
som = SOM(b=6, c=6, k=6)
som.fit(X)
bmus = som.transform(X)
plot_iris(bmus, df['species'])

# alternative plotting method
plot_grid(bmus, df['species'])

# plotting without species
plot_grid(bmus)

# compute quantization error by step for Iris data
errs = []
all_nsteps = list(range(10, 410, 20))
for nsteps in all_nsteps:
    som = SOM(b=6, c=6, nsteps=nsteps, learning_rate_decay=200, initial_learning_rate=0.25, random_state=0)
    som.fit(X)
    err = som.quantization_error(X)
    errs.append(err)
    
plt.plot(all_nsteps, errs)
plt.title('Quantization error by number of training steps')
plt.ylabel('quantization error')
plt.xlabel('number of steps')


    
    



    