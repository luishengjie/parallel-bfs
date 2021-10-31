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
from mpi4py import MPI
import time
from src.load_graph import get_graph, gen_balanced_tree

def parallel_dot_prod(comm, world, rank, a, b):
    if rank !=0:
        b = None
    b = comm.bcast(b, root=0)

    if world == 1:
        result = np.dot(a,b)
    else:
        if rank == 0:
            a_row = a.shape[0]
    
            if a_row >= world:
                
                split = np.array_split(a, world, axis=0)
        else:
            split = None

        split = comm.scatter(split, root=0)

        split = np.dot(split, b)
    
        data = comm.gather(split, root=0)

        if rank == 0:
            result = np.vstack(data)
            return result
        else:
            return None

def bfs_parallel2D_partitioning(A, target, comm, world, rank,s=0):
    """ @param A: Takes in the undirected adjacency matrix of a tree
        @return 1: Node found
        @return 0: Program Exit (rank==0), Node not found
        @retrun -1: Multiprocess Exit (rank!=0)
    
    """
    
    F = np.zeros(A.shape[0])
    P_i = np.zeros(A.shape[0])

    F[s] = 1
    P_i[s] = 1
    
    while np.sum(F)>0:
        # Print Position
        # print([i for i, e in enumerate(F) if e != 0])

        # Only perform computation when rank=0, first (parent) process 
        if rank == 0:
            t_i = parallel_dot_prod(comm, world, rank, A, np.reshape(F.T, (-1, 1))).astype(np.bool).astype(np.int)
            for i in range(A.shape[0]):
                if P_i[i]==1 and t_i[i]==1:
                    t_i[i] = 0

            # Check if search algo has found node
            for i, e in enumerate(t_i):
                if i==target and e==1:
                    return 1

            for i in range(A.shape[0]):
                if P_i[i]==0 and t_i[i]==1:
                    P_i[i] = 1

            F = t_i.copy()
        else:
            parallel_dot_prod(comm, world, rank, A, F.T)
    
    if rank==0:
        return 0
    else:
        return -1


def main():
    comm = MPI.COMM_WORLD
    world = comm.size
    rank = comm.Get_rank()
    start_time = time.time()
    G  = gen_balanced_tree(4, 3, directed=False)
    
    target = 999
    flag = bfs_parallel2D_partitioning(G, target, comm=comm, world=world, rank=rank)
    print("--- %s seconds ---" % (time.time() - start_time))
    if flag == 1:
        print(f"Node Found")
    else:
        print(f"Node not Found")

    if flag >= 0:
        # Since there are no serial sections in MPI code, 
        # a flag is used to terminate the MPI programme.
        comm.Abort()

if __name__=='__main__':
    main()