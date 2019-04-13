import time
import sys
import numpy as np
from numpy.random import RandomState
import tools

class MEP:
    def __init__(self,
                 N=100,     # number of nodes
                 L=1,       # number of layers   
                 K=2,       # number of communities
                 num_realisation=1,  # number of realisation
                 tolerance=0.1, # covergence tolerence
                 rseed=0,   # seed for random real numbers
                 out_adjacency=False,
                 infinity=1e10,
                 threshold=0.00001,
                 err=0.1,   # error added when initialising the parameters from file
                 undirected=False,
                 folder="data/",
                 adj="SocialMedia.txt", 
                 aff_file="aff.txt"):
        self.N = N
        self.L = L
        self.K = K
        self.num_realisation = num_realisation
        self.tolerance = tolerance
        self.rseed = rseed
        self.out_adjacency = out_adjacency
        self.infinity = infinity
        self.threshold = threshold
        self.err = err
        self.undirected = undirected
        self.folder = folder
        self.adj = adj
        self.aff_file = aff_file

        # Values for updating
        self.out = np.zeros((self.N, self.K), dtype=float)  # Matrix with nodes with only outgoing edges
        self.inc = np.zeros((self.N, self.K), dtype=float)  # Matrix with nodes with only incoming edges
        self.aff = np.zeros((self.K, self.K, self.L), dtype=float)  # Affinity matrix (similarity between 2 communities on each layer)

        # Old values of the matrix for comparing
        self.out_old = np.zeros((self.N, self.K), dtype=float)
        self.inc_old = np.zeros((self.N, self.K), dtype=float)
        self.aff_old = np.zeros((self.K, self.K, self.L), dtype=float)

        #  Final values that maximize Likelihood (convergence)
        self.out_f = np.zeros((self.N, self.K), dtype=float)
        self.inc_f = np.zeros((self.N, self.K), dtype=float)
        self.aff_f = np.zeros((self.K, self.K, self.L), dtype=float)
        
    # Intialise the affinity matrix with all the random values (Using the random_sample function)
    def randomiseAff(self, rng):
        for i in range(self.L):
            for k in range(self.K):
                for q in range(k, self.K):
                    if (q == k):
                        self.aff[k, q, i] = rng.random_sample(1)
                    else:
                        self.aff[k, q, i] = self.aff[ q, k, i] = self.err * rng.random_sample(1) # Why multiply

    def randomizeOutInc(self, rng, out_list, inc_list): # randomise the membership entries except from zero
        rng = np.random.RandomState(self.rseed) # random number generator
        for k in range(self.K):
            for i in range(len(out_list)):
                j = out_list[i]
                self.out[j][k] = rng.random_sample(1) # assign a random value for the node associated with an outgoing edge
                if (self.undirected == True):
                    self.inc[j][k] = self.out[j][k] # if the graph is undirected, then use the same random value for the nodes associated incoming and outgoing edge
            if (self.undirected == False):
                for i in range(len(inc_list)):
                    j = inc_list[i] # if the graph is directed, assign a differnt random value for the node associated with an incoming edge
                    self.inc[j][k] = rng.random_sample(1)

    # Function calling the randomise functions
    def initialize(self, out_list, inc_list, nodes):
        rng = np.random.RandomState(self.rseed) # RandomState is a method for generating random numbers drawn from probability distributions
        # Calling the randomise function
        self.randomiseAff(rng)
        self.randomizeOutInc(rng, out_list, inc_list)

    # display the affinity matrix
    def displayAffinity(self):  
        print(" aff:")
        for l in range(self.L):
            print("Layer: ", (l + 1))
            for k in range(self.K):
                for q in range(self.K):
                    print(self.aff[k][q][l])

    # update the old variables
    def updateOldVar(self, out_list, inc_list): # update the old variables
        for i in range(len(out_list)):
            for k in range(self.K):
                self.out_old[out_list[i]][k] = self.out[out_list[i]][k]
        for i in range(len(inc_list)):
            for k in range(self.K):
                self.inc_old[inc_list[i]][k] = self.inc[inc_list[i]][k]
        for l in range(self.L):
            for k in range(self.K):
                for q in range(self.K):
                    self.aff_old[k][q][l] = self.aff[k][q][l]

    # Function to copy the matrices to the old matrices
    def updateFinalParam(self):
        self.out_f = np.copy(self.out)
        self.inc_f = np.copy(self.inc)
        self.aff_f = np.copy(self.aff)

    # Display results after convergence
    def display(self, maxL, nodes):
        node_list = np.sort([int(i) for i in nodes])
        infile1 = self.folder + "out_K" + str(self.K) + ".txt"
        infile3 = self.folder + "aff_K" + str(self.K) + ".txt"
        # Open only the affinity and outgoing in write mode incase it is undirected
        in1 = open(infile1, 'w')
        in3 = open(infile3, 'w')
        print(in1, "Max Likelihood: ", maxL, "\nnum_realisation: ", self.num_realisation, "\n")
        print(in3, "Max Likelihood: ", maxL, "\nnum_realisation: ", self.num_realisation, "\n")
        if (self.undirected == False):
            infile2 = self.folder + "inc_K" + str(self.K) + ".txt"
            in2 = open(infile2, 'w') # Open the incoming file in write mode once we know that the graph is directed
            print("Max Likelihood: ", maxL, " \nnum_realisation: ", self.num_realisation, "\n", file=in2)

        # Print the node number and the final probability value of each node in the community
        for out in node_list:
            i = nodes.index(str(out))
            print("Node:", out, file=in1)
            if (self.undirected == False):
                print("Node:", out, file=in2)
            for k in range(self.K):
                print(self.out_f[i][k], file=in1)
                if (self.undirected == False):
                    print(self.inc_f[i][k], file=in2)
            print(file=in1)
            if (self.undirected == False):
                print(file=in2)

        # Close the files after writing to it
        in1.close()
        if (self.undirected == False):
            in2.close()

        # Print the probability of affinity of one community with another
        for l in range(self.L):
            print("Layer: ", (l + 1), file=in3)
            for k in range(self.K):
                for q in range(self.K):
                    print(self.aff_f[k][q][l], file=in3)
                print(file=in3)
            print(file=in3)
        in3.close()
        self.displayAffinity()

        # print the location of the output file
        print("Data saved in: ")
        print(infile1)
        print(infile3)
        if (self.undirected == False):
            print(infile2)


#  EM = Estimation Maximisation. Iterative method to find out the MAP

    # Update the node's value associated with the outgoing edge only 
    def updateOUT(self, A):

        # Derivative of out and aff_K
        D_out = np.einsum('iq->q', self.inc_old)
        aff_K = np.einsum('kqa->kq', self.aff_old)
        Z_outk = np.einsum('q,kq->k', D_out, aff_K)
        # Calculate the old value of rho (eq 5 numerator)
        rho_ijka = np.einsum('jq,kqa->jka', self.inc_old, self.aff_old)
        # Calculate the new value of rho (eq 5 numerator)
        rho_ijka = np.einsum('ik,jka->ijka', self.out, rho_ijka)
        # Eq 5 denominator
        Z_ija = np.einsum('ijka->ija', rho_ijka)
        Z_ijka = np.einsum('k,ija->ijka', Z_outk, Z_ija)

        non_zeros = Z_ijka > 0.
        
        # Final value of rho
        rho_ijka[non_zeros] /= Z_ijka[non_zeros]

        # New value of out, using the values of A and rho
        self.out = np.einsum('aij,ijka->ik', A, rho_ijka)

        # Initialise very less values to 0
        low_values_indices = self.out < self.threshold
        self.out[low_values_indices] = 0.
        # Difference between the new and the old values
        dist_out = np.amax(abs(self.out - self.out_old))
        self.out_old = self.out
        return dist_out


    # Update the node's value associated with the incoming edge only
    def updateINC(self, A):

        # Derivative of inc and aff_K
        D_inc = np.einsum('iq->q', self.out_old)
        aff_K = np.einsum('qka->qk', self.aff_old)
        Z_inck = np.einsum('q,qk->k', D_inc, aff_K)
        # Calculate the old value of rho (eq 5 numerator)
        rho_jika = np.einsum('jq,qka->jka', self.out_old, self.aff_old)
        # Calculate the new value of rho (eq 5 numerator)
        rho_jika = np.einsum('ik,jka->jika', self.inc, rho_jika)
        # Eq 5 denominator
        Z_jia = np.einsum('jika->jia', rho_jika)
        Z_jika = np.einsum('k,jia->jika', Z_inck, Z_jia)

        non_zeros = Z_jika > 0.

        # Final value of rho
        rho_jika[non_zeros] /= Z_jika[non_zeros]

        # New value of inc, using the values of A and rho
        self.inc = np.einsum('aji,jika->ik', A, rho_jika)

        # Initialise very less values to 0
        low_values_indices = self.inc < self.threshold
        self.inc[low_values_indices] = 0.
        # Difference between the new and the old values
        dist_inc = np.amax(abs(self.inc - self.inc_old))
        self.inc_old = self.inc

        return dist_inc

    # Update the value of the affinity matrix
    def updateAFF(self, A):

        # Partial derivative of out and inc
        out_k = np.einsum('ik->k', self.out)
        inc_k = np.einsum('ik->k', self.inc)

        # Multiply the values of k to get a K*K matrix
        Z_kq = np.einsum('k,q->kq', out_k, inc_k)
        # Summation of inc over q (Eq. 5 denominator)
        Z_ija = np.einsum('jq,kqa->jka', self.inc, self.aff_old)
        # Summation of out over k (Eq. 5 denominator)
        Z_ija = np.einsum('ik,jka->ija', self.out, Z_ija)

        # Transpose matrix A
        B = np.einsum('aij->ija', A)

        non_zeros = Z_ija > 0.

        # Eq. 10 preprocessing
        Z_ija[non_zeros] = B[non_zeros] / Z_ija[non_zeros]

        # Eq. 10 numerator
        rho_ijkqa = np.einsum('ija,ik->jka', Z_ija, self.out)
        rho_ijkqa = np.einsum('jka,jq->kqa', rho_ijkqa, self.inc)
        rho_ijkqa = np.einsum('kqa,kqa->kqa', rho_ijkqa, self.aff_old)

        self.aff = np.einsum('kqa,kq->kqa', rho_ijkqa, 1. / Z_kq)

        # Initialise very less values to 0
        low_values_indices = self.aff < self.threshold
        self.aff[low_values_indices] = 0.
        # Difference between the new and the old values
        dist_aff = np.amax(abs(self.aff - self.aff_old))
        self.aff_old = self.aff
        return dist_aff

    # Returns the partial derivative for values of out, inc and aff 
    def updateEM(self, B):
        d_out = self.updateOUT(B)
        if (self.undirected == True):
            self.inc = self.out
            self.inc_old = self.inc
            d_inc = d_out
        else:
            d_inc = self.updateINC(B)
        d_aff = self.updateAFF(B)

        return d_out, d_inc, d_aff

    # Iterative function
    def Likelihood(self, A):
        # Multiply out, inc and aff to calculate Poisson distribution's mean mu (Eq. 1)
        mu_ija = np.einsum('kql,jq->klj', self.aff, self.inc)
        mu_ija = np.einsum('ik,klj->lij', self.out, mu_ija)
        l = -mu_ija.sum() # Eq.3 RHS 2nd operand

        non_zeros = A > 0
        # Log likelihood function (Eq. 3 RHS)
        logM = np.log(mu_ija[non_zeros]) # log
        Alog = A[non_zeros] * logM # Eq.3 RHS 1st operand
        l += Alog.sum()

        if (np.isnan(l)):
            print("Likelihood is NaN")
            sys.exit(1)
        else:
            return l # Max Likelihood (Eq.3 LHS)

    # Function to check if the likelihood value has changed or not. Returns the number of iteration
    def checkConvergence(self, B, iter, new_L, coincide, convergence):
        if (iter % 10 == 0):
            old_L = new_L
            new_L = self.Likelihood(B)
            if (abs(new_L - old_L) < self.tolerance): # if (new likehood-old likelihood)>0, it coincides
                coincide += 1
            else:
                coincide = 0
        if (coincide > 10):
            convergence = True  # if it coincides more than 10 times, it converges
        iter += 1
        return iter, new_L, coincide, convergence

    # Main function of the program to initialize and perform the EM function
    def cycleRealizations(self, A, B, out_list, inc_list):
        maxL = -1000000000  # 1Billion
        nodes = list(A[0].nodes())
        for r in range(self.num_realisation):
            self.initialize(out_list, inc_list, nodes)
            self.updateOldVar(out_list, inc_list)
            coincide = 0
            convergence = False
            iter = 0
            new_L = self.infinity
            delta_out = delta_inc = delta_aff = self.infinity
            print("Updating r:", r, " ...")
            tic = time.clock()
            while (convergence == False and iter < 500):
                # Updates matrices and calculates the maximum difference between new and old
                delta_out, delta_inc, delta_aff = self.updateEM(B)

                iter, new_L, coincide, convergence = self.checkConvergence(
                    B, iter, new_L, coincide, convergence)
            print("r: ", r, " Likelihood: ", new_L, " iterations: ", iter,
                  ' time: ',
                  time.clock() - tic, 's')
            if (maxL < new_L):
                self.updateFinalParam()
                maxL = new_L
            self.rseed += 1
        print("Final Likelihood: ", maxL)
        self.display(maxL, nodes)
