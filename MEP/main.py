import sys
import time
from argparse import ArgumentParser

import networkx as nx
import numpy as np

import MEP as mep
import tools


def main():
    # Argument Passer
    infinity = 10000000000000 # 10 Trillion
    threshold = 0.0000001
    p = ArgumentParser()
    p.add_argument('-f', '--folder', type=str, default='')
    p.add_argument('-i', '--input', type=str, default='SocialMedia.txt')
    p.add_argument('-aff', '--aff_file', type=str, default='aff.txt')
    p.add_argument('-l', '--L', type=int, default=4)
    p.add_argument('-k', '--K', type=int, default=5)
    p.add_argument('-r', '--num_realisation', type=int, default=1)
    p.add_argument('-t', '--tolerance', type=float, default=0.1)
    p.add_argument('-e', '--err', type=float, default=0.1)
    p.add_argument('-u', '--undirected', type=int, default=0)
    p.add_argument('-s', '--rseed', type=int, default=0)
    args = p.parse_args()
    folder = "./data/" + args.folder
    
    if (args.undirected == True):
        A = [nx.MultiGraph() for l in range(args.L)] # For graphs
    else:
        A = [nx.MultiDiGraph() for l in range(args.L)] # For Directed Graphs

    tools.readGraph(folder, args.input, A)
    print("Undirected: ", bool(args.undirected))
    tools.printGraphStat(A, args.undirected)

    if (args.undirected == True):
        out_list = inc_list = tools.removeZeroEntriesUndirected(A) #  list of nodes with zero in and out degree
    else:
        out_list = tools.removeZeroEntriesOut(A) #  list of nodes with zero out degree
        inc_list = tools.removeZeroEntriesIn(A) #  list of nodes with zero in degree

    # Call to the EM function
    MEP = mep.MEP(
        N=A[0].number_of_nodes(),
        L=args.L,
        K=args.K,
        num_realisation=args.num_realisation,
        tolerance=args.tolerance,
        rseed=args.rseed,
        infinity=infinity,
        threshold=threshold,
        err=args.err,
        undirected=bool(args.undirected),
        folder=folder,
        input=args.input,
        aff_file=args.aff_file)

    # Start the clock
    startTimer = time.clock()
    N = A[0].number_of_nodes()  # Actual graph
    # print("@@@",A[0].nodes())
    # print("%%%",A[0].edges())
    B = np.empty(shape=[args.L, N, N])  # L*N*N matrix represntation of the graph
    # Populate the matrix B
    for l in range(args.L):
        B[l, :, :] = nx.to_numpy_matrix(A[l], weight='weight')

    MEP.cycleRealizations(A, B, out_list, inc_list)

    # Stop the clock
    stopTimer = time.clock()
    print(stopTimer - startTimer, " seconds.")

if __name__ == '__main__':
    main()
