import networkx as nx
import os
import numpy as np

# function to remove all the edges which has inward degree as zero for directed edges. 
def removeZeroEntriesIn(A):
    L = len(A)                  # number of layers
    zero_in = []
    nodes = list(A[0].nodes())
    for i in nodes:             # cycle over all the node
        k = 0
        for l in range(L):      # cycle over all the layers
            if (type(A[l].in_degree(i)) != dict):
                k += A[l].in_degree(i)
        if (k > 0):
            zero_in.append(nodes.index(i))  # append the nodes with degree more than 0
    return zero_in
    # returns the matrix without nodes with inward zero degree


# function to remove all the edges which has Outward degree as zero for directed edges.
def removeZeroEntriesOut(A):
    L = len(A)                      # number of layers
    zero_out = []
    nodes = list(A[0].nodes())
    for i in nodes:                 # cycle over all the nodes
        k = 0
        for l in range(L):          # cycle over all the layers
            if (type(A[l].out_degree(i)) != dict): k += A[l].out_degree(i)
        if (k > 0): zero_out.append(nodes.index(i))
    return zero_out
    # returns the matrix without nodes with outward degree as zero for directed edges.

# function to remove all the nodes which has zero degree, for undirected edges.
def removeZeroEntriesUndirected(A):
    L = len(A)                      # number of layers
    zero_in = []
    nodes = list(A[0].nodes())
    for i in nodes:                 # cycle over all the nodes
        k = 0
        for l in range(L):          # cycle over all the layers
            if (type(A[l].degree(i)) != dict): k += A[l].degree(i)
        if (k > 0): zero_in.append(nodes.index(i))
    return zero_in      
    # returns the matrix without nodes with degree as zero for undirected edges.

# Adds node i to all layers andr returns node index
def idx(i, A):
    L = len(A)      # number of layers
    if (i not in list(A[0].nodes())):
        for l in range(L):
            A[l].add_node(i)
            # returns nodes index


# function to read the graph from file and put it in correct format
def readGraph(folder, adjacency_file, A):
    print("Adjacency file :", folder + adjacency_file)
    infile = open(folder + adjacency_file, 'r') # Opening the file in read mode
    nr = 0
    L = len(A)
    for line in infile:
        a = line.strip('\n').split()
        if (a[0] == "E"):   # check whether the it starts with E i.e. EDGE
            if (nr == 0):
                l = len(a) - 3     # Skip the first 3 columns as they are E N1 N2
            v1 = a[1]
            v2 = a[2]
            idx(v1, A)
            idx(v2, A)
            for l in range(L):
                is_edge = int(a[l + 3])
                if (is_edge > 0):
                    A[l].add_edge(v1, v2, weight=is_edge)   # if edge exists add to the graph
    infile.close()

# Print the graph statistics
def printGraphStat(A, undirected=False):
    L = len(A)
    N = A[0].number_of_nodes()
    print("N=", N)
    for l in range(L):
        B = nx.to_numpy_matrix(A[l], weight='weight')   # Create the numpy matrix of A
        if undirected == False:
            E = np.sum(B)
        else:
            E = 0.5 * np.sum(B) # No of edges is half in case of undirected
        print('E[', (l + 1), ']:', E, " Density:",
              100 * float(E) / float(N * (N - 1)))


def outGraph(folder, A):    # function to write the output in a text file.
    L = len(A)
    for a in range(L):
        outfile = folder + "outAdjacency" + str(a + 1) + ".txt"
        outf = open(outfile, 'w')
        print("Adjacency of layer ", (a + 1), " output in: ", outfile)
        # Traverse the edges and add the nodes to the output file
        for e in A[a].edges():
            i = e[0]
            j = e[1]
            print(i, j, file=outf)
        outf.close()
