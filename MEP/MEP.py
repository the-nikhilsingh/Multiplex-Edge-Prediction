import time
import sys
import numpy as np
from numpy.random import RandomState
import tools

class MEP:
    def __init__(self,
                 N=100,     #number of nodes
                 L=1,       #number of layers   
                 K=2,       #number of communities
                 N_real=1,     
                 tolerance=0.1, #covergence tolerence
                 rseed=0,   #seed for random real numbers
                 out_adjacency=False,
                 infinity=1e10,
                 max_err=0.00001,
                 err=0.1,   #error added when initialising the parameters from file
                 undirected=False,
                 folder="data/",
                 adj="SocialMedia.txt", 
                 aff_file="aff.txt"):
        self.N = N
        self.L = L
        self.K = K
        self.N_real = N_real
        self.tolerance = tolerance
        self.rseed = rseed
        self.out_adjacency = out_adjacency
        self.infinity = infinity
        self.max_err = max_err
        self.err = err
        self.undirected = undirected
        self.folder = folder
        self.adj = adj
        self.aff_file = aff_file

        #Values for updating
        self.out = np.zeros((self.N, self.K), dtype=float)  #Matrix with nodes with only outgoing edges
        self.inc = np.zeros((self.N, self.K), dtype=float)  #Matrix with nodes with only incoming edges
        self.aff = np.zeros((self.K, self.K, self.L), dtype=float)  #Affinity matrix (similarity between 2 communities on each layer)

        #Old values of the matrix for comparing
        self.out_old = np.zeros((self.N, self.K), dtype=float)
        self.inc_old = np.zeros((self.N, self.K), dtype=float)
        self.aff_old = np.zeros((self.K, self.K, self.L), dtype=float)

        # Final values that maximize Likelihood (convergence)
        self.out_f = np.zeros((self.N, self.K), dtype=float)
        self.inc_f = np.zeros((self.N, self.K), dtype=float)
        self.aff_f = np.zeros((self.K, self.K, self.L), dtype=float)
        
    #Intialise the affinity matrix with all the random values (Using the random_sample function)
    def randomiseAff(self, rng):
        for i in range(self.L):
            for k in range(self.K):
                for q in range(k, self.K):
                    if (q == k):
                        self.aff[k, q, i] = rng.random_sample(1)
                    else:
                        self.aff[k, q, i] = self.aff[
                            q, k, i] = self.err * rng.random_sample(1)

    #Initalise the incoming and outgoing matrix
    def randomiseOutInc(self, rng, out_list, inc_list):
        rng = np.random.RandomState(self.rseed)

    def randomizeOutInc(self, rng, out_list, inc_list): #randomise the membership entries except from zero
        rng = np.random.RandomState(self.rseed) #random number generator
        for k in range(self.K):
            for i in range(len(out_list)):
                j = out_list[i]
                self.out[j][k] = rng.random_sample(1) #assign a random value for the node associated with an outgoing edge
                if (self.undirected == True):
                    self.inc[j][k] = self.out[j][k] #if the graph is undirected, then use the same random value for the nodes associated incoming and outgoing edge
            if (self.undirected == False):
                for i in range(len(inc_list)):
                    j = inc_list[i] #if the graph is directed, assign a differnt random value for the node associated with an incoming edge
                    self.inc[j][k] = rng.random_sample(1)

    #Function calling the randomise functions
    def initialize(self, out_list, inc_list, nodes):
        rng = np.random.RandomState(self.rseed) #RandomState is a method for generating random numbers drawn from probability distributions
        infile1 = self.folder + 'out_K' + str(self.K) + self.aff_file
        infile2 = self.folder + 'inc_K' + str(self.K) + self.aff_file
        aff_infile = self.folder + 'aff_K' + str(self.K) + self.aff_file
        #Calling the randomise function
        self.randomiseAff(rng)
        self.randomiseOutInc(rng, out_list, inc_list)

    #Function to display display the degree of the nodes
    def displayMembership(self, nodes):
        print(" out : ")
        for i in range(self.N):
            print(nodes[i])
            for k in range(self.K):
                print(self.out[i][k])
        if (self.undirected == False):
            print(" inc : ")
            for i in range(self.N):
                print(nodes[i])
                for k in range(self.K):
                    print(self.inc[i][k])

    #display the affinity matrix
    def displayAffinity(self):  
        print(" aff:")
        for l in range(self.L):
            print("Layer: ", (l + 1))
            for k in range(self.K):
                for q in range(self.K):
                    print(self.aff[k][q][l])

    #update the old variables
    def updateOlD_incar(self, out_list, inc_list):
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

    #Function to copy the matrices to the old matrices
    def updateFinalParam(self):
        self.out_f = np.copy(self.out)
        self.inc_f = np.copy(self.inc)
        self.aff_f = np.copy(self.aff)

    #Display results after convergence
    def display(self, maxL, nodes):
        node_list = np.sort([int(i) for i in nodes])
        infile1 = self.folder + "out_K" + str(self.K) + ".txt"
        infile3 = self.folder + "aff_K" + str(self.K) + ".txt"
        #Open only the affinity and outgoing in write mode incase it is undirected
        in1 = open(infile1, 'w')
        in3 = open(infile3, 'w')
        print(in1, "Max Likelihood: ", maxL, "\nN_real: ", self.N_real, "\n")
        print(in3, "Max Likelihood: ", maxL, "\nN_real: ", self.N_real, "\n")
        if (self.undirected == False):
            infile2 = self.folder + "inc_K" + str(self.K) + ".txt"
            in2 = open(infile2, 'w') #Open the incoming file in write mode once we know that the graph is directed
            print("Max Likelihood: ", maxL, " \nN_real: ", self.N_real, "\n", file=in2)

        #Print the node number and the final probability value of each node in the community
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

        #Close the files after writing to it
        in1.close()
        if (self.undirected == False):
            in2.close()

        #Print the probability of affinity of one community with another
        for l in range(self.L):
            print("Layer: ", (l + 1), file=in3)
            for k in range(self.K):
                for q in range(self.K):
                    print(self.aff_f[k][q][l], file=in3)
                print(file=in3)
            print(file=in3)
        in3.close()
        self.displayAffinity()

        #print the location of the output file
        print("Data saved in: ")
        print(infile1)
        print(infile3)
        if (self.undirected == False):
            print(infile2)


# EM = Estimation Maximisation. Iterative method to find out the MAP

    #Update the node's value associated with the outgoing edge only 
    def updateOUT(self, A):

        D_out = np.einsum('iq->q', self.inc_old)
        aff_K = np.einsum('kqa->kq', self.aff_old)
        Z_outk = np.einsum('q,kq->k', D_out, aff_K)
        rho_ijka = np.einsum('jq,kqa->jka', self.inc_old, self.aff_old)

        rho_ijka = np.einsum('ik,jka->ijka', self.out, rho_ijka)

        Z_ija = np.einsum('ijka->ija', rho_ijka)
        Z_ijka = np.einsum('k,ija->ijka', Z_outk, Z_ija)

        non_zeros = Z_ijka > 0.

        rho_ijka[non_zeros] /= Z_ijka[non_zeros]

        self.out = np.einsum('aij,ijka->ik', A, rho_ijka)

        low_values_indices = self.out < self.max_err
        self.out[low_values_indices] = 0.
        dist_out = np.amax(abs(self.out - self.out_old))
        self.out_old = self.out
        return dist_out


    #Update the node's value associated with the incoming edge only
    def updateINC(self, A):

        D_inc = np.einsum('iq->q', self.out_old)
        aff_K = np.einsum('qka->qk', self.aff_old)
        Z_inck = np.einsum('q,qk->k', D_inc, aff_K)
        rho_jika = np.einsum('jq,qka->jka', self.out_old, self.aff_old)

        rho_jika = np.einsum('ik,jka->jika', self.inc, rho_jika)

        Z_jia = np.einsum('jika->jia', rho_jika)
        Z_jika = np.einsum('k,jia->jika', Z_inck, Z_jia)
        non_zeros = Z_jika > 0.

        rho_jika[non_zeros] /= Z_jika[non_zeros]

        self.inc = np.einsum('aji,jika->ik', A, rho_jika)

        low_values_indices = self.inc < self.max_err
        self.inc[low_values_indices] = 0.
        dist_inc = np.amax(abs(self.inc - self.inc_old))
        self.inc_old = self.inc

        return dist_inc

    #Update the value of the affinity matrix
    def updateAFF(self, A):
        out_k = np.einsum('ik->k', self.out)
        inc_k = np.einsum('ik->k', self.inc)
        Z_kq = np.einsum('k,q->kq', out_k, inc_k)
        Z_ija = np.einsum('jq,kqa->jka', self.inc, self.aff_old)
        Z_ija = np.einsum('ik,jka->ija', self.out, Z_ija)
        B = np.einsum('aij->ija', A)
        non_zeros = Z_ija > 0.
        Z_ija[non_zeros] = B[non_zeros] / Z_ija[non_zeros]
        rho_ijkqa = np.einsum('ija,ik->jka', Z_ija, self.out)
        rho_ijkqa = np.einsum('jka,jq->kqa', rho_ijkqa, self.inc)
        rho_ijkqa = np.einsum('kqa,kqa->kqa', rho_ijkqa, self.aff_old)
        self.aff = np.einsum('kqa,kq->kqa', rho_ijkqa, 1. / Z_kq)
        low_values_indices = self.aff < self.max_err
        self.aff[low_values_indices] = 0.
        dist_aff = np.amax(abs(self.aff - self.aff_old))
        self.aff_old = self.aff
        return dist_aff

    #Returns the partial derivative for values of out, inc and aff 
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

    #Iterative function
    def Likelihood(self, A):
        mu_ija = np.einsum('kql,jq->klj', self.aff, self.inc)
        mu_ija = np.einsum('ik,klj->lij', self.out, mu_ija)
        l = -mu_ija.sum()
        non_zeros = A > 0
        logM = np.log(mu_ija[non_zeros])
        Alog = A[non_zeros] * logM
        l += Alog.sum()

        if (np.isnan(l)):
            print("Likelihood is NaN")
            sys.exit(1)
        else:
            return l

    #Function to check if the likelihood value has changed or not. Returns the number of iteration
    def checkConvergence(self, B, iter, l2, coincide, convergence):
        if (iter % 10 == 0):
            old_L = l2
            l2 = self.Likelihood(B)
            if (abs(l2 - old_L) < self.tolerance):
                coincide += 1
            else:
                coincide = 0
        if (coincide > 10):
            convergence = True
        iter += 1
        return iter, l2, coincide, convergence

    #Main function of the program to initialize and perform the EM function
    def cycleRealizations(self, A, B, out_list, inc_list):
        maxL = -1000000000  #1Billion
        nodes = list(A[0].nodes())
        for r in range(self.N_real):
            self.initialize(out_list, inc_list, nodes)
            self.updateOlD_incar(out_list, inc_list)
            coincide = 0
            convergence = False
            iter = 0
            l2 = self.infinity
            delta_out = delta_inc = delta_aff = self.infinity
            print("Updating r:", r, " ...")
            tic = time.clock()
            while (convergence == False and iter < 500):
                #Updates matrices and calculates the maximum difference between new and old
                delta_out, delta_inc, delta_aff = self.updateEM(B)

                iter, l2, coincide, convergence = self.checkConvergence(
                    B, iter, l2, coincide, convergence)
            print("r: ", r, " Likelihood: ", l2, " iterations: ", iter,
                  ' time: ',
                  time.clock() - tic, 's')
            if (maxL < l2):
                self.updateFinalParam()
                maxL = l2
            self.rseed += 1
        print("Final Likelihood: ", maxL)
        self.display(maxL, nodes)
