#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:36:25 2021

@author: Michaelkhalfin
"""

import networkx as nx
# import numpy as np
import matplotlib.pyplot as plt
# from pprint import pprint

G = nx.DiGraph()

# random small world network
def small_world(N, K, P):
    """ generates random small world network
    """
    G = nx.watts_strogatz_graph(n = N, k = K, p = P)
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
    G = small_world(500, 10, .13)
    visualize_network(G)