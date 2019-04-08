import networkx as nx
import os
import numpy as np


def removeZeroEntriesIn(A):     #function to remove all the edges which has inward degree as zero for directed edges. 
    L = len(A)                  #number of layers
    zero_in = []
    nodes = list(A[0].nodes())
    for i in nodes:             #cycle over all the node
        k = 0
        for l in range(L):      #cycle over all the layers
            if (type(A[l].in_degree(i)) != dict):
                k += A[l].in_degree(i)
        if (k > 0):
            zero_in.append(nodes.index(i))  #append the nodes with degree more than 0
    return zero_in              #returns the matrix without nodes with inward zero degree


def removeZeroEntriesOut(A):       #function to remove all the edges which has Outward degree as zero for directed edges.
    L = len(A)                      #number of layers
    zero_out = []
    nodes = list(A[0].nodes())
    for i in nodes:                 #cycle over all the nodes
        k = 0
        for l in range(L):          #cycle over all the layers
            if (type(A[l].out_degree(i)) != dict): k += A[l].out_degree(i)
        if (k > 0): zero_out.append(nodes.index(i))
    return zero_out             #returns the matrix without nodes with outward degree as zero for directed edges.


def removeZeroEntriesUndirected(A): #function to remove all the nodes which has zero degree, for undirected edges.
    L = len(A)                      #number of layers
    zero_in = []
    nodes = list(A[0].nodes())
    for i in nodes:                 #cycle over all the nodes
        k = 0
        for l in range(L):          #cycle over all the layers
            if (type(A[l].degree(i)) != dict): k += A[l].degree(i)
        if (k > 0): zero_in.append(nodes.index(i))
    return zero_in      #returns the matrix without nodes with degree as zero for undirected edges.


def idx(i, A):       #Adds node i to all layers andr returns node index
    L = len(A)      #number of layers
    if (i not in list(A[0].nodes())):
        for l in range(L):
            A[l].add_node(i)
            #returns nodes index


def readGraph(folder, adjacency_file, A):       #function to read the graph from file and put it in correct format
    print("Adjacency file :", folder + adjacency_file)
    infile = open(folder + adjacency_file, 'r')
    nr = 0
    L = len(A)
    for line in infile:
        a = line.strip('\n').split()
        if (a[0] == "E"):   #check whether the it starts with E i.e. EDGE
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
    infile.close()          #close the file


def printGraphStat(A, undirected=False):    #print the graph statistics
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


def outGraph(folder, A):    #function to write the output in a text file.
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
