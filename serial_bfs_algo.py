__author__ = "Lui Sheng Jie"
__email__ = "luishengjie@outlook.com"

""" Sequential BFS implementation.
    Sequential implementation of Algorithm 1 Parallel BFS algorithm: High-level overview [1].

    Reference: 
    [1] https://www.researchgate.net/publication/220782745_Scalable_Graph_Exploration_on_Multicore_Processors

"""


import numpy as np
import time
from src.load_graph import get_graph, gen_balanced_tree

def get_adjacent_nodes(G, x):
    idx_lst = []
    adj_list = G[x]
    for idx, val in enumerate(adj_list):
        if val == 1:
            idx_lst.append(idx)
    return idx_lst

def bfs_seq(G, target):
    r = 0
    CQ = []
    
    # Init all values in P to inf
    P = [np.inf for i in range(G.shape[0])]
    # Set root node 
    P[r] = 0
    
    # Enqueue r
    CQ.append(r)

    while len(CQ) != 0:
        # print(f"CQ: {CQ}")
        NQ = []
        
        for i in range(len(CQ)):
            # Dequeue CQ
            u = CQ.pop(0)
            # For each v adjacent to u
            for v in get_adjacent_nodes(G, u):
                if v == target:
                    return True
                if P[v] == np.inf:
                    P[v] = u
                    NQ.append(v)
        # Swap CQ and NQ
        tmp = NQ
        NQ = CQ
        CQ = tmp
    return False

def main():
    start_time = time.time()
    G  = gen_balanced_tree(5, 5, directed=True)
    print(G.shape)
    # G = get_graph()
    find_node = bfs_seq(G, target=999999)
    print("--- %s seconds ---" % (time.time() - start_time))
    if find_node:
        print(f"Node Found")
    else:
        print(f"Node not Found")


if __name__=='__main__':
    main()