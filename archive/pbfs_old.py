import time
import numpy as np
import multiprocessing
from multiprocessing.managers import BaseManager
from collections import deque

# This is thr vertex node
class Node(object):
    def __init__(self, val=0, left=None, right=None,child1=None,child2=None):
        self.elem = val
        self.left = left
        self.right = right
        self.child1 = child1
        self.child2 = child2
        self.dist = np.inf

# This is a bin_tree using chain storage, cause this alogrithem take use of chain storage, it's easy to convert adj_matrix to it
class BinTree(object):
    

    def __init__(self):
        self.root = None
        self.list = []

    def add(self, number):
        node = Node(number)

        if not self.root:
            self.root = node
            self.list.append(self.root)
        else:

            while True:
                point = self.list[0]
                if not point.child1:
                    point.child1 = node
                    self.list.append(point.child1)
                    return
                elif not point.child2:
                    point.child2 = node
                    self.list.append(point.child2)
                    self.list.pop(0)
                    return
def get_BinTree(k):
    temp = BinTree()
    for i in range(k):
        temp.add(i)
    return temp

def levelOrder(root):
    results = []
    if not root:
        return results

    from collections import deque
    que = deque([root])

    while que:
        size = len(que)
        result = []
        for _ in range(size):
            cur = que.popleft()
            result.append(cur.elem)
            if cur.child1:
                que.append(cur.child1)
            if cur.child2:
                que.append(cur.child2)
        # print(result)
        results.append(result)

    return results
def pennantorder(root):
    i=0
    
    # if not in_pennant.root:
    #     return results
    from collections import deque
    que = deque([root])
    while que:
        size = len(que)
        result = []
        for _ in range(size):
            cur = que.popleft()

            i=i+1

            result.append(cur.dist)
            if cur.left:
                que.append(cur.left)
            if cur.right:
                que.append(cur.right)
    # print(i)

# pennant and bag are special data_structure proposed by author, they are the core idea in this algo
class pennant(object):
    def __init__(self,k):
        self.root = None
        self.second = None

        self.size = 2**k
        self.queue = []

    def add(self, item):
        node = item
        if self.root is None:
            self.root = node
            return
        elif self.second is None:
            self.second = node
            self.root.left = self.second
            self.queue.append(self.second)

            return
        else:
           
            while True:
                
                cur_node = self.queue[0]
                
                if cur_node.left is None:
                    cur_node.left = node
                    self.queue.append(cur_node.left)
                    return
                # 判断右结点
                elif cur_node.right is None:
                    cur_node.right = node
                    self.queue.append(cur_node.right)
                    self.queue.pop(0)
                    return

    def split(self):
        y = pennant(int(np.log2(self.size//2)))
        y.root = self.root.left
        y.second = y.root.left

        self.second = self.second.right
        self.root.left = self.second
        y.root.right = None
        self.size = int(self.size/2)
        return y

def union(x,y):

    u=pennant(int(np.log2(x.size))+1)
    u.root = x.root
    u.second = y.root
    u.root.left = u.second
    u.second.right = x.second
    return u

class Bag(object):
    def __init__(self):
        self.baglist = [None for k in range(30)]
        self.empty = [None for k in range(30)]

    def __getitem__(self, item):
        return self.baglist[item]

    def insert(self,data):
        k=0
        temp = pennant(k)
        temp.add(data)
        while(self.baglist[k]!=None):
            temp=union(temp,self.baglist[k])
            self.baglist[k]=None
            k=k+1

        else:
            self.baglist[k]=temp

    def is_empty(self):
        if self.baglist==self.empty:
            return True
        else:
            return False

# The search consists of four function:pbfs,process_layer,process_pennant,travel_penant. The later is the children function of former
# The process_pennant is a recursion function, and parallelization will be used here
def travel_pennant(in_pennant,out_bag,d,target):
    
    que = deque([in_pennant.root])
    while que:
        size = len(que)
        result = []
        for _ in range(size):
            cur = que.popleft()
            # print(cur.elem)
            if cur.elem == target:
                return True
            if cur.child1:
                if cur.child1.dist==np.inf:
                    cur.child1.dist=d+1
                    out_bag.insert(cur.child1)
            if cur.child2:
                if cur.child2.dist == np.inf:
                    cur.child2.dist=d+1
                    out_bag.insert(cur.child2)
            if cur.left:
                que.append(cur.left)
            if cur.right:
                que.append(cur.right)

# parrallelization is applyed in this part. The two process will operate the same object:next_bag
def process_pennant(in_pennant,next_bag,d,target):
    if in_pennant.size < 128:
        if travel_pennant(in_pennant,next_bag,d,target):
            return True
    else:
        new_pennant = in_pennant.split()

        if process_pennant(in_pennant, next_bag, d,target):
            return True
        if process_pennant(new_pennant, next_bag, d,target):
            return True

    return False

def process_layer(cur_bag,next_bag,d,target):
    print([cur_bag.baglist[k] for k in range(30)])
    for k in range(30):
        if cur_bag.baglist[k] != None:
            if process_pennant(cur_bag.baglist[k],next_bag,d,target):
                return True

def pbfs(v0,target):
    v0.dist=0
    d=0
    cur_bag = Bag()
    cur_bag.insert(v0)
    while not cur_bag.is_empty():
        next_bag = Bag()
        if process_layer(cur_bag,next_bag,d,target):
            return True
        d=d+1
        cur_bag = next_bag

if __name__ == '__main__':
    b= get_BinTree(30)
    graph = get_BinTree(999)
    
    start_time = time.time()
    find_node=pbfs(graph.root, target=-1)
    print("--- %s seconds ---" % (time.time() - start_time))
    if find_node:
        print('Node found')
    else:
        print('Node not found')
    

