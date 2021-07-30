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
def small_world(N, k, P):
    """ generates random small world network
    """
    #G = nx.newman_watts_strogatz_graph(n = N, k = K, p = P)
    #G = nx.DiGraph.to_directed(G)
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

# random scale free network
# 1 <= m < n
def scale_free(N,m0):
    """ generates random scale free network
    """
    # G = nx.barabasi_albert_graph(n = N, m = M)
    # G = sf.make_sparse_adj_matrix(4)
    #E = 0
    #mat = np.zeros([N, N], dtype=int)
    # for i in range(1,m0):
    #     for j in range(i+1, m0):
    #         mat[i,j] = 1
    #         E += 2
    # for i in range(m0+1,N):
    #     current_deg = 0
    #     while current_deg < m:
    #         while True:
    #             loc = np.random.randint(0,n)
    #             if loc == i or mat[i, loc] == 1:
    #                 continue
    #             mat[i, loc] = 1
    #             break
    #         adj = 0
    #         for rep in mat[i]:
    #             if rep == 1:
    #                 adj += 1
    #         b = float(adj)/E
    #         chance = np.random.random()
    #         if b > chance:
    #             chance = np.random.random()
    #             mat[i,j] = 1
    #             E += 2
    #         else:
    #             mat[i,j] = 1
    #             E += 1
    #             no_connection = True
    #             while no_connection:
    #                 while True:
    #                     h = np.random.randint(0,n)
    #                     if h == i or mat[i, h] == 1:
    #                         continue
    #                     chance = np.random.random()
    #                     adj = 0
    #                     for rep in mat[i]:
    #                         if rep == 1:
    #                             adj += 1
    #                     b = adj/E
    #                     if b > chance:
    #                         mat[h,i] = 1
    #                         E += 1
    #                         no_connection = False
    #                     break
    return mat

def visualize_matrix(G):
    """ used to visualize the network
    """
    #N = nx.to_numpy_array(G)
    plt.imshow(G, cmap='gray')
    
def create_network(G):
    new = nx.from_numpy_matrix(G, create_using=nx.DiGraph())
    return new
    
def visualize_network(G):
    pos = nx.circular_layout(G)
    nx.draw(G, pos, node_size = 10, width = 0.1, arrowsize = 2)
    
if __name__ == "__main__":
    n = 50 # 1000
    k = 4 # 20
    
    Pvals = np.linspace(0,1,100)
    clusts = np.zeros_like(Pvals)
    pathlens = np.zeros_like(Pvals)
    for i, p, in enumerate(Pvals):
        mat = small_world(N = n,k = k,P = p)
        net = create_network(mat)
        clusts[i] = nx.average_clustering(net) # edit
        pathlens[i] = nx.average_shortest_path_length(net)
    
    # fig, ax = plt.subplots()
    # plt.title('Small world: n = {}, k = {}'.format(n,k))
    # plt.xlabel('p-values')
    # ax.scatter(Pvals, clusts, color='black', label='clustering coeff.')
    # ax.scatter(Pvals, pathlens, color='red', label='avg. path length')
    # ax.legend()
    # plt.show()
    
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_title('Small world: n = {}, k = {}'.format(n,k))
    ax2 = fig.add_subplot(2,1,2)
    ax2.set_xlabel('p-values')
    #ax1.set_yticks(np.limspace(0,1,5))
    ax1.scatter(Pvals, clusts, color='black', label='clustering coeff.')
    ax2.scatter(Pvals, pathlens, color='red', label='avg. path length')
    ax1.legend()
    ax2.legend()
    
    # Mvals = np.array(range(3,n))
    # clusts2 = np.zeros_like(Mvals)
    # pathlens2 = np.zeros_like(Mvals)
    # edges = np.zeros_like(Mvals)
    
    # for i, m, in enumerate(Mvals):
    #     #mat = scale_free(N = n, m0 = m)
    #     net = create_network(mat)
    #     clusts2[i] = nx.average_clustering(net) # edit
    #     pathlens2[i] = nx.average_shortest_path_length(net)
    #     edges[i] = nx.number_of_edges(net)
    
    # fig, ax = plt.subplots()
    # plt.title('Scale free: n = {}'.format(n))
    # plt.xlabel('m-values')
    # ax.scatter(Mvals, clusts, color='black', label='clustering coeff.')
    # ax.scatter(Mvals, pathlens, color='red', label='avg. path length')
    # ax.legend()
    # ax.legend()
    
    # fig = plt.figure(figsize=(8,8))
    # ax1 = fig.add_subplot(3,1,1)
    # ax1.set_title('Scale free: n = {}'.format(n))
    # ax2 = fig.add_subplot(3,1,2)
    # ax2.set_xlabel('m-values')
    # ax3 = fig.add_subplot(3,1,3)
    # ax1.scatter(Mvals, clusts2, color='black', label='clustering coeff.')
    # ax2.scatter(Mvals, pathlens2, color='red', label='avg. path length')
    # ax3.scatter(Mvals, edges, color='blue', label='number of edges')
    # ax1.legend()
    # ax2.legend()
    # ax3.legend()
    # plt.show()
    # G = small_world(500, 10, .13)
    # visualize_network(G)