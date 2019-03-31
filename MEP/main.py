import MEP as mep
import numpy as np
import networkx as nx
from argparse import ArgumentParser
import sys
import tools
import time


def main():
    infinity = 10000000000000
    max_err = 0.0000001
    p = ArgumentParser()
    p.add_argument('-f', '--folder', type=str, default='')
    p.add_argument('-a', '--adj', type=str, default='SocialMedia.txt')
    p.add_argument('-aff', '--aff_file', type=str, default='aff.txt')
    p.add_argument('-l', '--L', type=int, default=4)
    p.add_argument('-k', '--K', type=int, default=5)
    p.add_argument('-r', '--N_real', type=int, default=1)
    p.add_argument('-e', '--tolerance', type=float, default=0.1)
    p.add_argument('-g', '--err', type=float, default=0.1)
    p.add_argument('-o', '--out_adjacency', type=int, default=1)
    p.add_argument('-u', '--undirected', type=int, default=0)
    p.add_argument('-z', '--rseed', type=int, default=0)
    args = p.parse_args()

    folder = "../data/" + args.folder
    if (args.undirected == True):
        A = [nx.MultiGraph() for l in range(args.L)]
    else:
        A = [nx.MultiDiGraph() for l in range(args.L)]

    tools.readGraph(folder, args.adj, A)
    print("Undirected: ", bool(args.undirected))
    tools.printGraphStat(A, args.undirected)

    if (args.out_adjacency):
        tools.outGraph(folder, A)

    if (args.undirected == True):
        out_list = inc_list = tools.removeZeroEntriesUndirected(A)
    else:
        out_list = tools.removeZeroEntriesOut(A)
        inc_list = tools.removeZeroEntriesIn(A)

    MEP = mep.MEP(
        N=A[0].number_of_nodes(),
        L=args.L,
        K=args.K,
        N_real=args.N_real,
        tolerance=args.tolerance,
        rseed=args.rseed,
        out_adjacency=bool(args.out_adjacency),
        infinity=infinity,
        max_err=max_err,
        err=args.err,
        undirected=bool(args.undirected),
        folder=folder,
        adj=args.adj,
        aff_file=args.aff_file)

    tic = time.clock()
    N = A[0].number_of_nodes()
    B = np.empty(shape=[args.L, N, N])

    for l in range(args.L):
        B[l, :, :] = nx.to_numpy_matrix(A[l], weight='weight')

    MEP.cycleRealizations(A, B, out_list, inc_list)

    toc = time.clock()
    print(toc - tic, " seconds.")


if __name__ == '__main__':
    main()