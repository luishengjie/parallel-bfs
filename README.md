# parallel-bfs
Python implementation of parallel breadth-first search


# Run Experiments
## Parallel BFS implementation using 2D partitioning
- Parallel implementation: mpirun -np 2 python bfs_parallel_2d_mpi.py 
- Sequential implementation: python seq_bfs_2d_partition.py 

## Parallel BFS Using Queue
- Parallel implementation: mpirun -np 2 python parallel_bfs_algo.py 
- Sequential implementation: python seq_bfs_algo.py 

# Dependencies
- networkx
- mpi4py
