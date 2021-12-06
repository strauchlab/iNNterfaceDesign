import json, numpy as np, copy, string

def readjson(jobfile):
    with open(jobfile, 'r') as fl:
        data = json.load(fl)
    return data

def readfile(jobfile,type):
    inputfile = open(jobfile, "r")
    if type=='t':
        text = inputfile.read()
    if type=='l':
        text = inputfile.readlines()
    inputfile.close()
    return text

def keywords(ff):
    text = readfile(ff, 'l')
    text = {a.split(':')[0]: a.split(':')[1].strip() for a in text if (a[0] != '#') and (a.split(':')[1].strip() != '')}
    if 'pdb' not in text:
        print('PDB-file is not provided')
        exit()
    if 'prefix' not in text:
        text['prefix'] = text['pdb']
    if 'interf_type' not in text:
        text['interf_type'] = str(0)
    text['amn_design'] = 'PepSep1' if ('amn_design' not in text) else 'PepSep' + str(text['amn_design']) 
    return text

def writejson(jobfile, text):
    with open(jobfile, 'w') as fl:
        json.dump(text, fl)

def writefile(name, text):
    ff = open(name, 'w')
    ff.write(text)
    ff.close()

def one_hot_amns(seq):
    oha = np.zeros((len(seq), 20))
    for ix, i in enumerate(seq):
        oha[ix][i - 1] = 1
    return oha

def distance(i, j):
    d = ((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2 + (i[2] - j[2]) ** 2) ** 0.5
    return d

def angle(v):
    v1 = v[1] - v[0]
    v2 = v[2] - v[0]
    prod = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    tet = np.arccos(prod /(np.linalg.norm(v1) * np.linalg.norm(v2)))
    return tet

def dihedral(v):
    v1 = v[0] - v[1]
    v2 = v[1] - v[2]
    v3 = v[2] - v[3]
    n1 = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
    n2 = np.cross(v2, v3) / np.linalg.norm(np.cross(v2, v3))
    normv2 = v2 / np.linalg.norm(v2)
    m1 = np.cross(n1, normv2)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    theta = np.arctan2(y, x)
    return theta

def toint(j):
    j = j.strip()
    if j[0] != '-':
        if j.isdigit():
            j = int(j)
        else:
            j1 = int(j[:-1])
            j2 = [ix+1 for ix, i in enumerate(string.ascii_uppercase + string.ascii_lowercase) if j[-1] == i][0]/100 
            j = j1 + j2
    else:
        j = j[1:]
        if j.isdigit():
            j = int(j)
        else:
            j1 = int(j[:-1])
            j2 = [ix+1 for ix, i in enumerate(string.ascii_uppercase + string.ascii_lowercase) if j[-1] == i][0]/100
            j = j1 + j2
        j = -j
    return j


def rotation_matrix(angle, direction, point=None):
    sina = np.sin(angle)
    cosa = np.cos(angle)
    direction = direction/np.linalg.norm(direction)
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def rotate_dihedral(angle, direction, point, coords):
    M = rotation_matrix(angle, direction, point)
    u = np.full(len(coords), 1)
    u = u[:, np.newaxis]
    coords = np.append(coords, u, axis=1)
    coords = np.dot(M, np.transpose(coords))
    coords = np.transpose(coords[:-1])
    return coords

def kabsch(P, Q):
    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if (d):
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    P = np.dot(P, U)
    return [rmsd(P, Q), P, U]

def rmsd(V, W):
    D = len(V[0])
    N = len(V)
    rmsd = 0.0
    for v, w in zip(V, W):
        rmsd += sum([(v[i]-w[i])**2.0 for i in range(D)])
    return np.sqrt(rmsd/N)

def cmdscale(D):
    n = len(D)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -H.dot(D ** 2).dot(H) / 2
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    w, = np.where(evals > 0)
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]
    Y = V.dot(L)
    return Y, evals

def dist_matrix2coords(D):
    Y, evals = cmdscale(D)
    coords = Y[:, :3]
    coords -= coords[1]
    coords = align_point(coords[0], np.array([1, 0, 0]), coords, ortha=None)
    coords = align_point(coords[2], np.array([-1, 1, 0]), coords, ortha=np.array([1, 0, 0]))
    coords2 = copy.deepcopy(coords)
    coords2[:, -1] *= -1
    return [coords, coords2]

def align_point(p1, p2, atoms_xyz, c= None, ortha=None):
    c_old = np.zeros(3)
    if c is not None:
        p1 -= c
        p2 -= c
        atoms_xyz -= c
        c_old = copy.deepcopy(c)
    c= np.array([0, 0, 0])
    if ortha is None:
        norm = p1 / np.linalg.norm(p1)
        ortha = np.cross(p2, norm)
    a1 = dihedral(np.array([p1, c, ortha, p2]))
    if a1 == a1:
        atoms_xyz = rotate_dihedral(a1, ortha, c, atoms_xyz) + c_old
    return atoms_xyz

def make_ranges(data, k):
    if len(data) <= k:
        chunks = [data]
    else:
        ranges = list(range(0, len(data), k))
        if len(data) - ranges[-1] > k/2:
            ranges = [[ranges[i], ranges[i + 1]] for i in range(0, len(ranges) - 1)] + [[ranges[-1], len(data)]]
        else:
            ranges = [[ranges[i], ranges[i + 1]] for i in range(0, len(ranges) - 2)] + [[ranges[-2], len(data)]]
        chunks = [data[r[0]:r[1]] for rdx, r in enumerate(ranges)]
    return chunks

def neigh(xyz, r):
    res = []
    for x in range(-r, r+1):
        for y in range(-r, r+1):
            for z in range(-r, r+1):
                if abs(distance([x, y, z], np.array([0, 0, 0])) - r) <= 1:
                    res.append([x, y, z])
    return np.array(res) + np.array(xyz)

def sphere_coords(n1, n2):
    coords = []
    for x in range(-n1, n1 + 1):
        for y in range(-n1, n1 + 1):
            for z in range(-n1, n1 + 1):
                d = distance([x, y, z], [0, 0, 0])
                if (d <= n1) and (d>=n2):
                    coords.append([x, y, z])
    return coords

def separate_frags(pose):
    pocket_pose = pose.clone()
    pocket_pose.delete_residue_range_slow(1, 6)
    loop_pose = pose.clone()
    loop_pose.delete_residue_range_slow(7, pose.total_residue())
    return loop_pose, pocket_pose

def get_explicit_numbers(num):
    num = [a.strip() for a in num.split(',')]
    num = [[int(j) for j in i.split('-')] for i in num]
    num = [[int(i[0])] if len(i) == 1 else list(range(i[0], i[1]+1)) for i in num]
    return num

def get_coords(pose):
    coords = []
    for i in range(1, pose.total_residue()+1):
        coords_i = []
        num = pose.residue(i).nheavyatoms()
        for j in range(1, num +1):
            coords_i.append(np.array(pose.residue(i).xyz(j)).tolist())
        coords.append(coords_i)
    return coords
