__author__ = "Lui Sheng Jie"
__email__ = "luishengjie@outlook.com"

""" Parallel BFS implementation using 2D partitioning [1]

    Reference: 
    [1]. "Parallel breadth-first search on distributed memory systems" 
        by Buluç, Aydin, and Kamesh Madduri. International Conference 
        for High Performance Computing, Networking, Storage and Analysis, 2011

"""

import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
import time
from src.load_graph import get_graph, gen_balanced_tree

def dot_prod(X, Y):
    return np.dot(X, Y.T).astype(np.bool).astype(np.int)

def bfs_parallel2D_partitioning(A, target, s=0):
    """ @param A: Takes in the undirected adjacency matrix of a tree
        @return True: Node found
        @return False: Node not found
    """
    
    F = np.zeros(A.shape[0])
    P_i = np.zeros(A.shape[0])

    F[s] = 1
    P_i[s] = 1
    
    while np.sum(F)>0:
        # Print Position
        # print([i for i, e in enumerate(F) if e != 0])
        t_i = dot_prod(A, F.T)
                
        for i in range(A.shape[0]):
            if P_i[i]==1 and t_i[i]==1:
                t_i[i] = 0
        
        # Check if search algo has found node
        for i, e in enumerate(t_i):
            if i==target and e==1:
                return True

        for i in range(A.shape[0]):
            if P_i[i]==0 and t_i[i]==1:
                P_i[i] = 1
        F = t_i.copy()

    return False

def main():
    start_time = time.time()
    G  = gen_balanced_tree(2, 2, directed=False)
    # G = get_graph(directed=False)

    find_node = bfs_parallel2D_partitioning(G, target = 10)
    print("--- %s seconds ---" % (time.time() - start_time))

    if find_node:
        print(f"Node Found")
    else:
        print(f"Node not Found")

if __name__=='__main__':
    main()