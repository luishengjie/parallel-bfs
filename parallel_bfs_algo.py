__author__ = "Lui Sheng Jie"
__email__ = "luishengjie@outlook.com"

""" Baseline parallel BFS implementation.
    Algorithm 1 Parallel BFS algorithm: High-level overview [1] was implemented.

    Reference: 
    [1] https://www.researchgate.net/publication/220782745_Scalable_Graph_Exploration_on_Multicore_Processors

"""

import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
from mpi4py import MPI

import time
from src.load_graph import get_graph, gen_balanced_tree
from functools import partial



def get_adjacent_nodes(G, x):
    idx_lst = []
    adj_list = G[x]
    for idx, val in enumerate(adj_list):
        if val == 1:
            idx_lst.append(idx)
    return idx_lst

# def get_neighbour(u, G, target):
#     nq = []
#     # For each v adjacent to u
#     # print(u)
#     found_node = False
#     for v in get_adjacent_nodes(G, u):
#         if v == target:
#             found_node = True
#         if P_ARR[v] == np.inf:
#             P_ARR[v] = u
#             nq.append(v)
#     return nq, found_node

def get_neighbour(u_list, G, p_arr, target, comm, world, rank):
    split = np.array_split(u_list, world, axis=0)
    split = comm.scatter(split, root=0)
    found_node = False
    nq = []
    for u in split:
        for v in get_adjacent_nodes(G, u):
            if v == target:
                found_node = True
            if p_arr[v] == np.inf:
                p_arr[v] = u
                nq.append(v)

    data = comm.gather(nq, root=0)
    if rank == 0:
        result = []
        for d in data:
            result += d
        return result, found_node, p_arr
        

def bfs_parallel(G, target, comm, world, rank):
    """ @param A: Takes in the directed adjacency matrix of a tree
        @return 1: Node found
        @return 0: Program Exit (rank==0), Node not found
        @retrun -1: Multiprocess Exit (rank!=0)
    
    """
    r = 0
    cq = []
    p_arr = []
    # Init all values in P to inf
    for i in range(G.shape[0]):
        p_arr.append(np.inf)

    # Set root node 
    p_arr[r] = 0

    # Enqueue r
    cq.append(r)
    while len(cq) != 0:
        # print(f"CQ: {CQ}")
        # Parallel Dequeue
        if rank == 0:
            nq, found_node, p_arr = get_neighbour(cq, G, p_arr, target, comm, world, rank)
            if found_node:
                # print(f"NQ: {NQ}")
                return 1
            # Swap CQ and NQ
            cq = nq
        else:
            get_neighbour(cq, G, p_arr, target, comm, world, rank)
    if rank==0:
        return 0
    else:
        return -1


def main():
    comm = MPI.COMM_WORLD
    world = comm.size
    rank = comm.Get_rank()

    start_time = time.time()
    G  = gen_balanced_tree(4, 5, directed=True)
    target = 999999
    # G = get_graph()
    flag = bfs_parallel(G, target=target, comm=comm, world=world, rank=rank)
    
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