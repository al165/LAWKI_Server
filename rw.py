from tqdm import tqdm

import numpy as np
import numpy.random as rand
import pandas as pd

from itertools import cycle

from sklearn.neighbors import KDTree, BallTree


def cosine_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosine_dist(a, b):
    return 1- cosine_sim(a, b)

def euclid_dist(a, b):
    return np.linalg.norm(a-b)

def euclid_dist(a, b):
    return ((a-b)**2).sum()



class MetaRW:
    def __init__(self, rw_class, n_walks, **rw_kwargs):
        self.walkers = [rw_class(**rw_kwargs) for _ in range(n_walks)]
        
    def step(self, i):
        samples = [w.step(i) for w in self.walkers]
        inds, rows, durations = list(zip(*samples))
        return inds, rows, durations
        


class RW:
    abs_min = 4
    abs_max = 15
    
    @staticmethod
    def load_data(directory, is_2D):
        if is_2D:
            space = np.loadtxt(directory+"embeddings_tsne.tsv", delimiter="\t")
        else:
            space = np.loadtxt(directory+"embeddings.tsv", delimiter="\t")
        
        meta = pd.read_csv(directory+"meta.csv")
        return space, meta

    
    def __init__(self, is_2D, directory="./", space=None, meta=None):
        
        if space is None and meta is None:
            self.space, self.meta = RW.load_data(directory, is_2D)
        else:
            self.space, self.meta = space, meta
            
        self.tree = KDTree(self.space)
        self.n = self.space.shape[0]
        
        
        
    def sample_duration(self, row):
        secs = row.duration
        if secs < RW.abs_min:
            return 0, secs

        return self.sample_duration_normal(secs)
    
    
    def sample_duration_unif(self, secs):
        start = secs*0.3 # WADSWORTH CONSTANT
        max_length = secs - start - (secs*0.1)
        return start, rand.uniform(min(RW.abs_min, max_length), min(max_length, RW.abs_max))

        
        
    def sample_duration_normal(self, secs):
        start = secs*0.3 # WADSWORTH CONSTANT
        
        length = secs - start - (secs*0.1)
        m = (min(length, RW.abs_max) - RW.abs_min)/2
        s = rand.normal(m, scale=2)
        return start, min(RW.abs_max, abs(s)+RW.abs_min)

        
class Line2D(RW):    
    def __init__(self, n_neighbours=40, hist_len=5, restart_percentage=0.3,
                 directory="./", space=None, meta=None):
        super().__init__(is_2D=True, directory=directory, space=space, meta=meta)

        self.n_neighbours = n_neighbours
        self.hist = -hist_len
        self.restart_percentage = restart_percentage

        self.init = rand.randint(self.n)
        self.cur = self.init
        
        self.sampled = [self.cur]
        self.sampling_pool = set(range(self.n)) - {self.cur}
        
    def get_neighbours(self, cur_index, n_neighbours=None):
        if n_neighbours is None:
            n_neighbours = self.n_neighbours
        return self.tree.query(self.space[cur_index].reshape(1, -1), n_neighbours)[1][0, 1:]
    
    
    
    def step_(self, i, n_neighbours, ref, cur_dist):
#         if n_neighbours is None:
#             n_neighbours = self.n_neighbours
        
        neighs = self.get_neighbours(self.cur, n_neighbours=n_neighbours)
        neighs = list(filter(lambda n: n in self.sampling_pool, neighs))
#         print(f"1. {neighs=}")
        neighs = [n for n in neighs if euclid_dist(self.space[ref], self.space[n]) > cur_dist]
#         print(f"2. {neighs=}")
        new = neighs[rand.randint(len(neighs))]
        
        return new
    
    
    def step(self, i):
        ref = self.sampled[0] if self.hist < 0 else self.sampled[ref_i]
        cur_dist = euclid_dist(self.space[ref], self.space[self.cur])
        
        new = None
        n_neighbours = self.n_neighbours
        
        while new is None:
            try:
                new = self.step_(i, n_neighbours, ref, cur_dist)
            except ValueError:
                n_neighbours = n_neighbours*2
                print(f"{n_neighbours=}")
                if n_neighbours > self.n:
                    new = rand.choice(tuple(self.sampling_pool))
                
        self.cur = new
        
        self.sampled.append(self.cur)
        self.sampling_pool.remove(self.cur)
        if len(self.sampling_pool) < self.n*self.restart_percentage:
            print(f"{len(self.sampling_pool)=}")
            self.sampling_pool = set(range(self.n)) - set(self.sampled[-100:])
        
        cur_row = self.meta.iloc[self.cur]
        
        return self.cur, cur_row, self.sample_duration(cur_row)    
    
    
#     def step(self, i):        
#         ref = self.sampled[0] if self.hist < 0 else self.sampled[ref_i]
#         cur_dist = euclid_dist(self.space[ref], self.space[self.cur])

        
#         try:
#             neighs = self.get_neighbours(self.cur)
#             neighs = list(filter(lambda n: n in self.sampling_pool, neighs))
#             neighs = [n for n in neighs if euclid_dist(self.space[ref], self.space[n]) > cur_dist]
#             self.cur = neighs[rand.randint(len(neighs))]
            
#         except ValueError:
#             neighs = self.get_neighbours(self.cur, self.n_neighbours*100)
#             neighs = list(filter(lambda n: n in self.sampling_pool, neighs))
#             neighs = [n for n in neighs if euclid_dist(self.space[ref], self.space[n]) > cur_dist]
#             self.cur = neighs[rand.randint(len(neighs))]

# #             self.cur = rand.randint(self.n)
# #             self.cur = rand.choice(tuple(self.sampling_pool))
# #             cur = sampled[move_back]
# #             move_back -= 1
        
#         self.sampled.append(self.cur)
#         self.sampling_pool.remove(self.cur)
#         if len(self.sampling_pool) < self.n*0.3:
#             print(f"{len(self.sampling_pool)=}")
#             self.sampling_pool = set(range(self.n)) - set(self.sampled[-100:])
        
#         cur_row = self.meta.iloc[self.cur]
        
#         return self.cur, cur_row, self.sample_duration(cur_row)
        

    
    
class NeighbourRW(RW):
    def __init__(self, n_neighbours=100, directory="./", space=None, meta=None):
        super().__init__(is_2D=True, directory=directory, space=space, meta=meta)
        
        self.n_neighbours = n_neighbours
        
        self.cur = rand.randint(self.n)
        self.sampled = [self.cur]
        self.sampling_pool = set(range(self.n)) - {self.cur}
                           
    
    
    def get_neighbours(self, cur_index, n_neighbours=None):
        if n_neighbours is None:
            n_neighbours = self.n_neighbours
        return self.tree.query(self.space[cur_index].reshape(1, -1), 
                               n_neighbours)[1][0, 1:]

    
    def step_(self, i, n_neighbours):
        neighs = self.get_neighbours(self.cur, n_neighbours)
#         print(f"{len(self.sampling_pool)=}")
        neighs = list(filter(lambda n: n in self.sampling_pool, neighs))
        new = rand.choice(neighs)
        
        return new

    
    
    def step(self, i):
        new = None
        n_neighbours = self.n_neighbours
        
        while new is None:
            try:
                new = self.step_(i, n_neighbours)
            except ValueError:
                n_neighbours = n_neighbours*2
                print(f"{n_neighbours=}")
                if n_neighbours > self.n:
                    new = rand.choice(tuple(self.sampling_pool))
        self.cur = new
        
        
        self.sampled.append(self.cur)
        self.sampling_pool.remove(self.cur)
        
        if len(self.sampling_pool) < self.n*0.3:
            self.sampling_pool = set(range(self.n)) - set(self.sampled[-100:])

        cur_row = self.meta.iloc[self.cur]
        
        return self.cur, cur_row, self.sample_duration(cur_row)    

        
        
        
class DecoyRW(RW):
    def __init__(self, n_neighbours=100, directory="./", space=None, meta=None):
        super().__init__(is_2D=True, directory=directory, space=space, meta=meta)
        
        self.meta = list(self.meta.links)
        
#         self.n_neighbours = n_neighbours
        
#         self.cur = rand.randint(self.n)
#         self.sampled = [self.cur]
#         self.sampling_pool = set(range(self.n)) - {self.cur}
                               
            
    def step(self, i):
        self.cur = rand.randint(self.n)
        
        cur_row = self.meta[self.cur]

        return self.cur, cur_row, self.sample_duration(cur_row)
    
    
class MyRow:
    def __init__(self):
        self.link = "zwzmSv67uM8"
        self.views = 13
    
class NoRW:
    def __init__(self, directory=None):
        self.cur = MyRow()
    
    def step(self, i):
        return 0, self.cur, (0, 1)
