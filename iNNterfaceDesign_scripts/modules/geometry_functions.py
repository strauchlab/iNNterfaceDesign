import numpy as np
from modules import functions as f

flat = lambda x: [a for b in x for a in b]

class Vertex(object):
    def __init__(self, number, coord):
        self.number = number
        self.coord = coord

    def find_edges(self,main_edges):
        self.edges = [[x[0], x[1]] for x in main_edges if self.number in x]
        self.aux = [x for j in self.edges for x in j if x != self.number]
        self.aux_conn = [[x[0], x[1]] for x in main_edges if (x[0] in self.aux) and (x[1] in self.aux)]

    def find_planes(self):
        edges = self.edges
        aux_conn = self.aux_conn
        tri = []
        for x in edges:
            for y in aux_conn:
                if x[1] == y[0]:
                    tri.append(x+[y[1]])
                elif x[0] == y[1]:
                    tri.append(y + [x[1]])
                elif (x[0] == y[0]) and ([x[1], y[1]] in edges):
                    tri.append(x+[y[1]])
                elif (x[0] == y[0]) and ([y[1], x[1]] in edges):
                    tri.append([x[0], y[1], x[1]])
                elif (x[1] == y[1]) and ([x[0], y[0]] in edges):
                    tri.append([x[0], y[0], x[1]])
                elif (x[1] == y[1]) and ([y[0], x[0]] in edges):
                    tri.append([y[0]]+x)
        tri_red = []
        for j in tri:
            if j not in tri_red:
                tri_red.append(j)
        tri_red = [i[1:] + [i[0]] if i.index(self.number) == 1 else i for i in tri_red]
        tri_red = [[i[2]] + i[:2] if i.index(self.number) == 2 else i for i in tri_red]
        self.tri = tri_red

def icosahedron(i, n, k):
    v = np.array([[0, 0, 1.9],
     [1.618, -0.52, 0.85],
    [1, 1.376, 0.85],
    [0, 0, -1.9],
    [-1.618, 0.52, -0.85],
    [-1, -1.376, -0.85],
    [0, -1.7, 0.85],
    [1.618, 0.52, -0.85],
    [-1, 1.376, 0.85],
    [0, 1.7, -0.85],
    [-1.618, -0.52, 0.85],
    [1, -1.376, -0.85]])

    def split(v, p):
        edges = find_edges(v, p[0])
        triang = make_poly(v, edges)
        v_add1 = np.array([v[t].mean(axis=0) for t in triang])
        v_add1 = v_add1 + v_add1 / p[1] * p[2]
        v = np.vstack((v, v_add1))
        return v

    if n>=1:
        v = split(v, [2.2, 1.5, 0.38])
        if n>=2:
            v = split(v, [1.5, 1.75, 0.15])
            if n == 3:
                v = split(v, [1.0, 1.85, 0.045])
    v *= k
    v += i
    #print(max(flat([[f.distance(i, j) for i in v] for j in v])))
    return v

def find_edges(v, maxd):
    edges = []
    for idx, i in enumerate(v):
        for jdx, j in enumerate(v):
            if idx != jdx:
                d = f.distance(i, j)
                if d < maxd:
                    k = [idx, jdx]
                    k.sort()
                    if k not in edges:
                        edges.append(k)
    return edges

def make_poly(v, edges):
    vertices = [Vertex(i, v[i]) for i in range(0, v.shape[0])]
    triang = []
    for i in vertices:
        i.find_edges(edges)
        i.find_planes()
        for j in i.tri:
            j.sort()
            if j not in triang:
                triang.append(j)
    return triang

def find_triang(pts, n):
    if n ==0:
        l = 3.3
    elif n == 1:
        l = 2.3
    elif n == 2:
        l = 1.6
    elif n == 3:
        l = 1.0
    edges = find_edges(pts, l)
    pts = np.array(pts)
    triang = make_poly(pts, edges)
    return triang

def neighs_dict(subset, atoms):
    neighs = {}
    for adx, a in enumerate(subset):
        dist = [f.distance(x, a) for x in atoms]
        neighs[adx] = [kx for kx, k in enumerate(dist) if (k <= 5) and (k > 0.1)]
    return neighs

def find_core(subset, atoms, atom_neighs):
    ni, k = 1, 1.63
    icos1 = icosahedron(np.array([0, 0, 0]), 0, k)
    icos2 = icosahedron(np.array([0, 0, 0]), ni, k)
    surf, core = [], []
    for ax, a in enumerate(subset):
        v = icos1 + a
        ds = [min([f.distance(atoms[i],j) for i in atom_neighs[ax]]) for j in v]
        ds = len([i for i in ds if i <= 3.11])
        if  icos1.shape[0] - ds == 0:
            core.append(ax)
        else:
            v = icos2 + a
            ds = [min([f.distance(atoms[i], j) for i in atom_neighs[ax]]) for j in v]
            ds = [ix for ix, i in enumerate(ds) if i > 3.11]
            if len(ds) <= 3:
                core.append(ax)
    return core
