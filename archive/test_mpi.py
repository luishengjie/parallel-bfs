from mpi4py import MPI
import numpy as np

def pp(a, comm, world, rank):
    split = np.array_split(a, world, axis=0)
    split = comm.scatter(split, root=0)
    print(rank, split)
    data = comm.gather(split, root=0)
    if rank==0:
        result = np.vstack(data)
        print(result)
    return rank, rank==world


comm = MPI.COMM_WORLD
world = comm.size
rank = comm.Get_rank()
a = [1,2,3,4,5,6]
x,y = pp(a, comm, world, rank)
print(x,y)



