import sys
import pandas as pd
from enum import Enum, auto
from collections import deque

class Label(Enum):
    UNASSIGN = auto()
    ASSIGN = auto()
    NOISE = auto()


class PointObj():
    def __init__(self, p):
        self.oid = p[1]
        self.x = p[2]
        self.y = p[3]
        self.label = Label.UNASSIGN
    
    def _dist(self, q): # euclidean distance with point p
        return ((self.x - q.x) ** 2 + (self.y - q.y) ** 2) ** 0.5
    
    def scanNeighbors(self, X, eps):
        neighbors = []
        for q in X:
            if q != self and self._dist(q) <= eps:
                neighbors.append(q)

        return neighbors
    
    def pushCluster(self, clusterSet, k):
        self.label = Label.ASSIGN
        if k in clusterSet:
            clusterSet[k].append(self.oid)
        else:
            clusterSet[k] = [self.oid]

def dbscan(X, eps, minpts, clusterSet):
    k = -1
    for p in X:
        if p.label == Label.UNASSIGN: # no visited
            neighbors = p.scanNeighbors(X, eps) # get neighbors in eps
            
            if len(neighbors) >= minpts: # this point is core
                k += 1
                p.pushCluster(clusterSet, k)
                
                queue = deque(neighbors)
                while queue:
                    n = queue.popleft()

                    if n.label != Label.ASSIGN:
                        n.pushCluster(clusterSet, k)
                        
                        neighbors = n.scanNeighbors(X, eps)
                        if len(neighbors) >= minpts: # this neighbor is also core
                            queue.extend(neighbors)
            else:
                p.label = Label.NOISE




if __name__ == '__main__':
    # command line arguments
    input_file_name = sys.argv[1]
    n = int(sys.argv[2])
    Eps = int(sys.argv[3])
    MinPts = int(sys.argv[4])

    # open files
    input_file = pd.read_csv('./data-3/' + input_file_name, '\t', header=None, names=['object_id', 'x', 'y'])

    X = [PointObj(item) for item in input_file.itertuples()]
    clusterSet = {}
    dbscan(X, Eps, MinPts, clusterSet)

    clusters = sorted(clusterSet.values(), key=lambda x:len(x))
    while len(clusters) > n:
        del clusters[0]

    for i in range(n):
        fn = input_file_name.rstrip('.txt')
        pd.DataFrame(sorted(clusters[i])).to_csv(f'{fn}_cluster_{i}.txt', header=False, index=False)


