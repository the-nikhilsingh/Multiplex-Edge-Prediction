# MEP (Multiplex Edge Prediction)
Multilayer network factorization, for community detection, link prediction and measure layer interdependence.

## Input format.
The multilayer adjacency matrix should be formatted as an edge list with L+3 columns:

`E node1 node2 3 0 0 1`

The first columns tells the algorithm that the row denotes an edge; the second and third are the source and target nodes of that edge, respectively; l+3 column tells if there is that edge in the l-th layer and the weigth (must be integer). In this example the edge node1 --> node2 exists in layer 1 with weight 3 and in layer 4 with weight 1, but not in layer 2 and 3.

## Output.
Three files will be generated inside the `data` folder: the two NxK membership matrices `OUT` and `INC`, and the KxK layer affinity matrix `AFF`. Supposing that K=4 and `E=".dat"` the output files will be inside `data` folder with names:
- `out_K4.dat`
- `inc_K4.dat`
- `aff_K4.dat`

The first line outputs the Max Likelihood among the realizations.
For the membership files, the subsequent lines contain L+1 columns: the first one is the node label, the follwing ones are the (not normalized) membership vectors' entries.
For the affinity matrix file, the subsequent lines start with the number of the layer and then the matrix for that layer.
For the restricted assortative version only the diagonal entries of the affinity matrix are printed. The first entry of each row is the layer index.

## Dependencies:
Needs three main Python modules to be downloaded:

* `numpy` : https://docs.scipy.org/doc/numpy-1.10.1/user/install.html
* `networkx` : https://networkx.github.io/documentation/development/install.html
* `argparse` : https://pypi.python.org/pypi/argparse

## What's included:
- `main.py` : General version of the algorithm. Considers both directed and undirected weigthed multilayer networks with any community structures (non-diagonal or restricted diagonal affinity matrices W).
- `MEP.py` : Contains the class definition of a Multilayer network with all the member functions required.
- `tools.py` : Contains non-class functions.

Use the version that most resembles your network, i.e. if you have an undirected network set the flag '-u=1'. If you also know that the partition is assortative then use the flag '-A=1'.

## How to run run the code:
`python main.py`

## Required arguments

- `-a` : Adjacency matrix file
- `-f` : Folder where the adjacency input and output are/will be stored (inside `data` folder).

## Optional arguments
* `-aff` : End of the file where the parameters can be initialized from, in case initialization variable is greater than 0.
* `-l` : Number of layers, default is 4.
* `-k` : Number of communities, default is 5.
* `-r` : Number of different realizations, the final parameters will be the one corresponding to the realization leading to the max likelihood. Default is 1.
* `-t` : Convergence tolerance. Default is 0.1 .
* `-e` : Error added when intializing parameters from file. Default is 0.1 .
* `-o` : Flag to output adjacency matrix. Default is 0 (False).
* `-s` : Seed for random real numbers.
* `-u` : Flag to call the undirected network, default is 0 (False).
