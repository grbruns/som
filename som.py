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

def eucl_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


def learning_rate(s, initial=0.1, k=100):
    """ Decrease the learning rate by a factor of 2 every k steps. """
    
    return initial * (.5**(s/k))


def neighbor_fun_gaussian(x1, x2, s, c1=0.1, c2=0.4):
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


def neighbor_fun_gaussian_alt(x1, x2, s, c1=0.4, c2 = 4.0, k=25):
    """ This version more closely follows Kohonen's 1990 paper. """
    
    d = np.linalg.norm(np.array(x1) - np.array(x2))
    c1, c2 = (0.5, 4.0)  # smallest, largest sigma values
    decay = .5**(s/k)    # cut in half every k steps
    sigma = c1 + (c2 - c1) * decay
    return np.exp(-(d**2)/sigma**2)


def neighbor_fun_triangular(x1, x2, s, b_init=4, c_init=4, k=100):
    """ Return a value between 0 and 1 reflecting the degree to
    which vectors x1 and x2 are neighbors.
    
    Reduce the neighborhood size by a factor of 2 every k steps. """
    
    d = np.abs(np.array(x1) - np.array(x2))   # distances in x and y
    decay = .5**(s/k)
    b = 1 + (b_init-1)*decay
    c = 1 + (c_init-1)*decay
    y = max(0, 1 - d[0]/b - d[1]/c)
    return y
    

class SOM:
    
    """ A simple implementation of self-organizing maps.
    A 2-dimensional output space is assumed. 
    """
    
    def __init__(self, b=4, c=4, nsteps=1000, k=3, 
                 initial_learning_rate=0.5, learning_rate_decay=100, random_state=None):
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
                    delta = neighbor_fun_gaussian((i,j), bmu, s) * rate * (x - self.W[i,j])
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

    bins = [ int(bmus[:,i].max() + 1) for i in [0,1] ]
    plt.figure(figsize=(3,3))
    df1 = pd.DataFrame(bmus, columns = ['x1', 'x2'])
    if hue is None:
        sns.displot(df1, x='x1', y='x2', bins=bins); 
    else:
        sns.displot(df1, x='x1', y='x2', hue=hue, bins=bins)
        
    xwidth, ywidth = [ (nbins-1)/nbins for nbins in bins ]
    plt.xticks([ xwidth * (i+0.5) for i in range(bins[0]) ], labels=range(bins[0]))
    plt.yticks([ ywidth * (i+0.5) for i in range(bins[1]) ], labels=range(bins[1]))
    
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
        ys.append(neighbor_fun_gaussian(x1, x2, s))
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
som = SOM(b=6, c=6, nsteps=200)
som.fit(X)
bmus = som.transform(X)
plot_iris(bmus, df['species'])

# test 4
som = SOM(b=6, c=6, k=6)
som.fit(X)
bmus = som.transform(X)
# plot_iris(bmus, df['species'])
plot_grid(bmus, df['species'])

# test 5
som = SOM(b=10, c=10)
som.fit(X[:,[1]]) # sepal width
bmus = som.transform(X[:,[1]]) 
plot_grid(bmus, df['species'])

# plotting without species
plot_grid(bmus)

# compute quantization error by step for Iris data
errs = []
all_nsteps = list(range(10, 510, 20))
for nsteps in all_nsteps:
    som = SOM(b=6, c=6, nsteps=nsteps, learning_rate_decay=100, initial_learning_rate=0.25, random_state=0)
    som.fit(X)
    err = som.quantization_error(X)
    errs.append(err)
    
plt.plot(all_nsteps, errs)
plt.title('Quantization error by number of training steps')
plt.ylabel('quantization error')
plt.xlabel('number of steps')

# =============================================================================
# example with 2D input space; plot location of neurons
# =============================================================================

import time

def plot_neurons_in_grid(W):
    """ Plot first two values of W in 2D space """

    plt.scatter(W[:,:,0], W[:,:,1])
    # for each W[i,j] in W, plot line from W[i,j] to W[i+1, j+1]
    for i in range(b):
        for j in range(c):
            if (i+1) < b:
                plt.plot([W[i,j,0], W[i+1,j,0]], [W[i,j,1], W[i+1,j,1]], color='black')
            if (j+1) < c:
                plt.plot([W[i,j,0], W[i,j+1,0]], [W[i,j,1], W[i,j+1,1]], color='black')

# take a sample of the data
sample = np.random.choice(X.shape[0], 100, replace=False) 
Xs = X[sample]
print(Xs[:3])

b,c = 5,5
for n in range(0, 610, 25):
    som = SOM(b=b, c=c, nsteps=n, initial_learning_rate=0.5, learning_rate_decay=100, random_state=0)
    som.fit(Xs)
    plt.figure(figsize=(4,4))
    plt.scatter(Xs[:,0], Xs[:,1], color="red", s=20)
    plot_neurons_in_grid(som.W)
    plt.title(f'step {n}')
    plt.show();
    time.sleep(0.25)


#
# neighborhood function test following
#

