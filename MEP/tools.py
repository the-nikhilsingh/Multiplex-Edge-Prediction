import networkx as nx
import os
import numpy as np


def removeZeroEntriesIn(A):
    L = len(A)
    zero_in = []
    nodes = list(A[0].nodes())
    for i in nodes:
        k = 0
        for l in range(L):
            if (type(A[l].in_degree(i)) != dict):
                k += A[l].in_degree(i)
        if (k > 0):
            zero_in.append(nodes.index(i))
    return zero_in


def removeZeroEntriesOut(A):
    L = len(A)
    zero_out = []
    nodes = list(A[0].nodes())
    for i in nodes:
        k = 0
        for l in range(L):
            if (type(A[l].out_degree(i)) != dict): k += A[l].out_degree(i)
        if (k > 0): zero_out.append(nodes.index(i))
    return zero_out


def removeZeroEntriesUndirected(A):
    L = len(A)
    zero_in = []
    nodes = list(A[0].nodes())
    for i in nodes:
        k = 0
        for l in range(L):
            if (type(A[l].degree(i)) != dict): k += A[l].degree(i)
        if (k > 0): zero_in.append(nodes.index(i))
    return zero_in


def idx(i, A):
    #Adds node i to all layers andr eturns node index
    L = len(A)
    if (i not in list(A[0].nodes())):
        for l in range(L):
            A[l].add_node(i)


def readGraph(folder, adjacency_file, A):
    print("Adjacency file :", folder + adjacency_file)
    infile = open(folder + adjacency_file, 'r')
    nr = 0
    L = len(A)
    for line in infile:
        a = line.strip('\n').split()
        if (a[0] == "E"):
            if (nr == 0):
                l = len(a) - 3
            v1 = a[1]
            v2 = a[2]
            idx(v1, A)
            idx(v2, A)
            for l in range(L):
                is_edge = int(a[l + 3])
                if (is_edge > 0):
                    A[l].add_edge(v1, v2, weight=is_edge)
    infile.close()


def printGraphStat(A, undirected=False):
    L = len(A)
    N = A[0].number_of_nodes()
    print("N=", N)
    for l in range(L):
        B = nx.to_numpy_matrix(A[l], weight='weight')
        if undirected == False:
            E = np.sum(B)
        else:
            E = 0.5 * np.sum(B)
        print('E[', (l + 1), ']:', E, " Density:",
              100 * float(E) / float(N * (N - 1)))


def outGraph(folder, A):
    L = len(A)
    for a in range(L):
        outfile = folder + "outAdjacency" + str(a + 1) + ".txt"
        outf = open(outfile, 'w')
        print("Adjacency of layer ", (a + 1), " output in: ", outfile)
        for e in A[a].edges():
            i = e[0]
            j = e[1]
            print(i, j, file=outf)
        outf.close()
