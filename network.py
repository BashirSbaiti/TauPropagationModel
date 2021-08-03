#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 19:31:03 2021

@author: Michaelkhalfin
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def small_world(N, k, P):
    """ generates random small world network
    """
    if N < 1:
        raise ValueError("N must be > 0")
    if k % 2 != 0:
        raise ValueError("k must be even")
    if P < 0 or P > 1:
        raise ValueError("P must be >= 0 and <= 1")
        
    mat = np.zeros([N, N], dtype=int)
    for i in range(N):
        for j in range(i-k//2, i+k//2 + 1):
            if j == i:
                continue
            ind = j % N
            mat[i, ind] = 1
            
    if P > 0:
        for i in range(N):
            for j in range(i-k//2, i+k//2 + 1):
                if j == i:
                    continue
                ind = j % N
                if np.random.random() <= P:
                    mat[i, ind] = 0
                    while True:
                        loc = np.random.randint(0, N)
                        if loc == i or mat[i, loc] == 1 or loc == ind:
                            continue
                        mat[i, loc] = 1
                        break
    return mat

def visualize_matrix(G):
    """ visualizes adjacency matrix
    """
    plt.imshow(G, cmap='gray')
    
def create_network(G):
    """ creates network from adjacency matrix
    """
    new = nx.from_numpy_matrix(G, create_using=nx.DiGraph())
    return new

def visualize_network(G):
    """ visualizes network
    """
    pos = nx.circular_layout(G)
    nx.draw(G, pos, node_size = 10, width = 0.1, arrowsize = 2)
    
if __name__ == "__main__":
    N = 1000
    k = 20
    
    Pvals = np.linspace(0,1,100)
    clusts = np.zeros_like(Pvals)
    pathlens = np.zeros_like(Pvals)
    for i, P, in enumerate(Pvals):
        mat = small_world(N = N,k = k,P = P)
        net = create_network(mat)
        clusts[i] = nx.average_clustering(net) # edit
        pathlens[i] = nx.average_shortest_path_length(net)
        
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_title('Small World Network: N = {}, k = {}'.format(N,k))
    ax2 = fig.add_subplot(2,1,2)
    ax2.set_xlabel('Probability (P)')
    ax1.scatter(Pvals, clusts, color='black', label='Mean Clustering Coefficient')
    ax2.scatter(Pvals, pathlens, color='red', label='Mean Path Length')
    ax1.legend()
    ax2.legend()