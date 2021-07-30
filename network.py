#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:36:25 2021

@author: Michaelkhalfin
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# from pprint import pprint

G = nx.DiGraph()

# random small world network
def small_world(N, K, P):
    """ generates random small world network
    """
    G = nx.connected_watts_strogatz_graph(n = N, k = K, p = P)
    return G

# random scale free network
# 1 <= m < n
def scale_free(N, M):
    """ generates random scale free network
    """
    G = nx.barabasi_albert_graph(n = N, m = M)
    return G

def visualize_network(G):
    """ used to visualize the network
    """
    pos = nx.circular_layout(G)
    nx.draw(G, pos, node_size = 10, width = 0.1, arrowsize = 2)

    # creates adjacency matrix
    N = nx.to_numpy_array(G)
    plt.imshow(N, cmap='gray')
    
if __name__ == "__main__":
    n = 500 # 1000
    k = 16 # 20
    
    Pvals = np.linspace(0,1,100)
    clusts = np.zeros_like(Pvals)
    pathlens = np.zeros_like(Pvals)
    for i, p, in enumerate(Pvals):
        net = small_world(N = n,K = k,P = p)
        clusts[i] = nx.average_clustering(net) # edit
        pathlens[i] = nx.average_shortest_path_length(net)
    
    fig, ax = plt.subplots()
    plt.title('Small world: n = {}, k = {}'.format(n,k))
    plt.xlabel('p-values')
    ax.scatter(Pvals, clusts, color='black', label='clustering coeff.')
    ax.scatter(Pvals, pathlens, color='red', label='avg. path length')
    ax.legend()
    
    Mvals = np.array(range(1,n))
    for i, m, in enumerate(Mvals):
        net = scale_free(N = n,M = m)
        clusts[i] = nx.average_clustering(net) # edit
        pathlens[i] = nx.average_shortest_path_length(net)
    
    fig, ax = plt.subplots()
    plt.title('Scale free: n = {}'.format(n))
    plt.xlabel('m-values')
    ax.scatter(Mvals, clusts, color='black', label='clustering coeff.')
    ax.scatter(Mvals, pathlens, color='red', label='avg. path length')
    ax.legend()
    # G = small_world(500, 10, .13)
    # visualize_network(G)