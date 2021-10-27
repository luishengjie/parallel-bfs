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
import time
from src.load_graph import get_graph, gen_balanced_tree
from functools import partial



P_ARR = []

def get_adjacent_nodes(G, x):
    idx_lst = []
    adj_list = G[x]
    for idx, val in enumerate(adj_list):
        if val == 1:
            idx_lst.append(idx)
    return idx_lst

def get_neighbour(u, G, target):
    nq = []
    # For each v adjacent to u
    # print(u)
    found_node = False
    for v in get_adjacent_nodes(G, u):
        if v == target:
            found_node = True
        if P_ARR[v] == np.inf:
            P_ARR[v] = u
            nq.append(v)
    return nq, found_node

def bfs_parallel(G, target):
    r = 0
    CQ = []
    
    # Init all values in P to inf
    for i in range(G.shape[0]):
        P_ARR.append(np.inf)

    # Set root node 
    P_ARR[r] = 0

    # Enqueue r
    CQ.append(r)
    while len(CQ) != 0:
        print(f"CQ: {CQ}")
        # Parallel Dequeue
        num_cpu = mp.cpu_count()
        with Pool(num_cpu) as pool:
            results = pool.map(partial(get_neighbour, G=G, target=target), CQ)
            nq_tmp = [x for (x,y) in results]
            for (x,y) in results:
                if y:
                    return True  
        # print(nq_tmp)
        NQ = list(np.concatenate(nq_tmp).ravel())
    
        # Swap CQ and NQ
        CQ = NQ

    return False


def main():
    start_time = time.time()
    G  = gen_balanced_tree(3, 4, directed=True)
    # G = get_graph()
    find_node = bfs_parallel(G, target=10000)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    if find_node:
        print(f"Node Found")
    else:
        print(f"Node not Found")

if __name__=='__main__':
    main()