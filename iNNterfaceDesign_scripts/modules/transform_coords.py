import numpy as np, copy, json, os
from numpy import sqrt, dot, cross
from numpy.linalg import norm
from itertools import combinations
from modules import functions as f

np.set_printoptions(suppress=True)
BQi = lambda x: [[x, 0, 0], [0, x, 0], [0, 0, x], [-x, 0, 0], [0, -x, 0], [0, 0, -x]]

sst_dict = {0: 'H', 1: 'E', 2: 'L'}

def st_or(xyz, bq, p):
    res = None
    xyz_tr = xyz - xyz[0]
    norm = xyz_tr[1]/np.linalg.norm(xyz_tr[1])
    ortha = np.cross(bq[1], norm)
    a1 = f.dihedral(np.array([xyz_tr[1], bq[0], ortha, bq[1]]))
    if a1 == a1:
        xyz_r1 = f.rotate_dihedral(a1, ortha, bq[0], xyz_tr)
        a2 = f.dihedral(np.array([xyz_r1[2], bq[0], xyz_r1[1], bq[2]]))
        if a2 == a2:
            xyz_r2 = f.rotate_dihedral(a2, xyz_r1[1], bq[0], xyz_r1)
            q = trilaterate(xyz_r2, p)
            if q is not None:
                pts1 = np.vstack((xyz_r2, q[0]))
                pts1 = f.rotate_dihedral(-a2, pts1[1], bq[0], pts1)
                pts1 = f.rotate_dihedral(-a1, ortha, bq[0], pts1) + xyz[0]
                pts2 = np.vstack((xyz_r2, q[1]))
                pts2 = f.rotate_dihedral(-a2, pts2[1], bq[0], pts2)
                pts2 = f.rotate_dihedral(-a1, ortha, bq[0], pts2) + xyz[0]
                res = [pts1[-1], pts2[-1]]
    return res

def trilaterate(pts, ds):
    temp1 = pts[1]-pts[0]
    e_x = temp1/norm(temp1)
    temp2 = pts[2]-pts[0]
    i = dot(e_x, temp2)
    temp3 = temp2 - i*e_x
    e_y = temp3/norm(temp3)
    e_z = cross(e_x,e_y)
    d = norm(pts[1]-pts[0])
    j = dot(e_y,temp2)
    x = (ds[0]**2 - ds[1]**2 + d*d) / (2*d)
    y = (ds[0]**2 - ds[2]**2 - 2*i*x + i*i + j*j) / (2*j)
    temp4 = ds[0]**2 - x*x - y*y
    if temp4<0:
        #raise Exception("The three spheres do not intersect!");
        ds[:2] *=1.01
        temp1 = pts[1]-pts[0]
        e_x = temp1/norm(temp1)
        temp2 = pts[2]-pts[0]
        i = dot(e_x, temp2)
        temp3 = temp2 - i*e_x
        e_y = temp3/norm(temp3)
        e_z = cross(e_x,e_y)
        d = norm(pts[1]-pts[0])
        j = dot(e_y,temp2)
        x = (ds[0]**2 - ds[1]**2 + d*d) / (2*d)
        y = (ds[0]**2 - ds[2]**2 - 2*i*x + i*i + j*j) / (2*j)
        temp4 = ds[0]**2 - x*x - y*y
        #if temp4<0:
        #    raise Exception("The three spheres do not intersect!");
    if temp4 >= 0:
        z = temp4**0.5
        res = [[x, y, z], [x, y, -z]]
    else:
        res = None
    return res

def check_side_of_plane(p, x):
    v1 = p[1] - p[0]
    v2 = p[2] - p[0]
    va = x - p[0]
    cp = np.cross(v1,v2)
    d = np.dot(cp, va)
    return d


def convert_dm2xyz_48o(dm, dm_ca, xx):
    BQ = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    dist = []
    coords = []
    for a in range(len(dm)):
        ind = np.argsort([abs(x-a) for x in range(len(dm))])[:4] 
        ind_d = [i for i in list(range(len(dm))) if i not in ind]
        init_pts = [dm_ca[ix] for ix in ind]
        ypd = np.array([dm[a][ix] for ix in ind])
        pos = []
        for ip in range(len(init_pts)):
            p1, yp1 = init_pts[ip], ypd[ip]
            pr, ypr = np.delete(np.array(init_pts), ip, 0), np.delete(ypd, ip)
            p = st_or(pr, BQ, ypr)
            if p is not None:
                #d = [np.abs(yp1 - f.distance(p1, i)) for i in p]
                d = [np.abs(yp1 - f.distance(p1, i)) for i in p]
                #if abs(d[0] - d[1]) > 0.7:
                p = p[np.argsort(d)[0]].tolist()
                pos.append(p)
        if pos != []:
            pos = np.mean(np.array(pos),axis=0)
            coords.append(pos.tolist())
        else:
            print('None')
            coords.append(None)   
    return coords


def convert_dm2xyz_48f(dm, x):
    BQ = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    dist = []
    init_pts1 = BQi(x)
    n = 0
    for a in range(len(dm)-2):
        if a==0:
            ind = np.argsort(dm[a])[:4]
            ind_d = np.argsort(dm[a])[4:5]
        else:
            ind = np.argsort(dm[a][:-a])[:4-a].tolist() + list(range(len(dm[a])-a, len(dm[a])))
            ind_d = [i for i in list(range(len(dm[a]))) if i not in ind]
        if n >0:
            continue
        
        init_pts_all = init_pts1[:len(BQi(1))-a] + init_pts1[len(BQi(1)):]
        init_pts = [init_pts_all[ix] for ix in ind]
        ypd = np.array([dm[a][ix] for ix in ind])
        p1d, yp1d = [init_pts_all[x] for x in ind_d], [dm[a][x] for x in ind_d]
        pos, pos2, pos3 = [], [], []
        for ip in range(len(init_pts)):
            p1, yp1 = init_pts[ip], ypd[ip]
            pr, ypr = np.delete(np.array(init_pts), ip, 0), np.delete(ypd, ip)
            p = st_or(pr, BQ, ypr)
            p1x, yp1x = [p1] + p1d, [yp1] + yp1d
            if p is not None:
                #d = [np.abs(yp1 - f.distance(p1, i)) for i in p]
                d = np.array([[np.abs(yp1x[x] - f.distance(p1x[x], i)) for i in p] for x in range(len(p1x))])
                d = d.sum(axis=0)
                #if abs(d[0] - d[1]) > 0.7:
                p = p[np.argsort(d)[0]].tolist()
                if (a > 0) and (ind[ip] in list(range(len(dm[a])-min(a,3), len(dm[a])))):
                    pos2.append(p)
                else:
                    pos.append(p)
        if pos != []:
            pos = np.mean(np.array(pos),axis=0)
            init_pts1.append(pos.tolist())
        else:
            if pos2 != []:
                pos = np.mean(np.array(pos2[:2]),axis=0)
                init_pts1.append(pos.tolist())
            else:
                n += 1
    coords = np.array(init_pts1[6:])
    #if (n == 0):

    #else:
    #    coords = None
    return coords

def align_geoms(D, g1):
    cp = lambda x: x.mean(axis=0)
    pg = f.dist_matrix2coords(D)
    c1 = cp(g1)
    #c1 = cp(g1[1:3])
    g = g1 - c1
    pg = [i - cp(i[1:5]) for i in pg]
    #pg = [i - cp(i[2:4]) for i in pg]

    pts1 = [np.array([cp(pg[k][3:5]), cp(pg[k][1:3]), cp(pg[k][2:4])]) for k in range(2)]
    pts2 = np.array([cp(g[2:]), cp(g[:2]), cp(g[1:3])])
    pg = [np.dot(pg[k], f.kabsch(pts1[k], pts2)[2]) for k in range(2)]

    #pg = [f.align_point(cp(i[3:5]), cp(g[2:]), i) if f.angle([cp(i[3:5]), np.zeros(3), cp(g[2:])]) > 0.05 else i for i in pg] 
    #pg = [f.align_point(cp(i[1:3]), cp(g[:2]), i, ortha = cp(g[2:])) if f.angle([cp(i[1:3]), np.zeros(3), cp(g[:2])]) > 0.05 else i for i in pg]
    pg = [i + c1 for i in pg]
    pg = [np.dot(pg[k], f.kabsch(pg[k][1:-1], g1)[2]) for k in range(2)]
    return pg

def align_geoms1(g1, g2):
    cp = lambda x: x.mean(axis=0)
    c1 = g1[1:].mean(axis=0)
    c2 = g2[:2].mean(axis=0)
 
    c = cp(np.array([c1, c2]))
    g1 -= c1 - c
    g2 -= c2 - c
    return g1, g2

def make_pdb(name, pdb_name_f, sh_atoms, rank, prefix, e = None, c_names = None):
    atoms = ['N ', 'CA', 'C ', 'O ']

    def check_clashes(l_ca):
        l_clashes = [sorted([f.distance(l_ca[a1], l_ca[a2]) for a1 in range(len(l_ca)) if abs(a1-a2)>=2])[0] for a2 in range(len(l_ca))]
        k = len([a for a in l_clashes if a <= 3.8])
        return k
 

    def make_string(ch_label, coord):
        #print(coord)
        string = 'ATOM    235  CA  GLY A  32       9.716  36.424  72.347  1.00  1.0            C\n'
        n1, n2 = 1, 1
        coords = []
        for idx in range(len(coord[0])):
            for jdx in range(len(coord)):
                n2_v = "{:7.0f}".format(n2)
                if coord[jdx][idx] is not None:
                    #print(coord[jdx][idx])
                    xyz_str = ''.join(["{:8.3f}".format(x) for x in coord[jdx][idx]])
                    n1s = "{:2}".format(n1)
                    xyz_pdb_a = string[:4] + n2_v + string[11:24] + n1s + string[27:31] + xyz_str + string[54:]
                    xyz_pdb_a = xyz_pdb_a.replace('CA', atoms[jdx])
                    xyz_pdb_a = xyz_pdb_a.replace('C\n', atoms[jdx][:1] + '\n') 
                    coords.append(xyz_pdb_a)
                n2 += 1
            n1 += 1
        coords = [i[:21] + ch_label + i[22:] for i in coords]
        return ''.join(coords)

    def get_old_c(xyz, pep_l):
        xyz_c = []
        for x2 in atoms:
            xyz_a = []
            for x1 in range(1, pep_l+1):
                l =[a for a in xyz if (int(a.split()[5]) == x1) and (a.split()[2] == x2.strip())]        
                if l == []:
                    xyz_a.append(None)
                else:
                    l = l[0]
                    xyz_a.append([float(a) for a in [l[30:38], l[38:46], l[46:54]]])

            xyz_c.append(np.array(xyz_a))
        return xyz_c
    
    def stack(a1, a2):
        a1 = [x if x is not None else None for x in a1]
        a2 = [x if x is not None else None for x in a2]
        a = a1 + a2
        return a

    ch_label = [i for i in name[len(prefix)+1:] if i.isdigit() == False][0]
    if e !=10:
        if e is None:
            pdb_name = '_'.join([name] + rank) + '.pdb'
            xyz_new = make_string(ch_label, sh_atoms)
            f.writefile(pdb_name_f + pdb_name, xyz_new)
            pdb_names = [pdb_name]
        else:
            pdb_names = []
            if e == 0:
                pdb_name = '_'.join([name] + rank + ['N']) + '.pdb'
                xyz_old = f.readfile(pdb_name_f[:-1] + '_id/' + name + '_id.pdb', 'l')
                xyz_old =  [a for a in xyz_old if a.startswith('ATOM') and (a.split()[2] in [k.strip() for k in atoms])]
                pep_l = int(xyz_old[-1].split()[5])
                xyz_old = get_old_c(xyz_old, pep_l)
                sh_atoms = [stack(sh_atoms[k][:3], xyz_old[k]) for k in range(4)]
            elif e == 1:
                pdb_name = '_'.join([name] + rank + ['C']) + '.pdb'
                xyz_old = f.readfile(pdb_name_f[:-1] + '_id/' + name + '_id.pdb', 'l')
                xyz_old =  [a for a in xyz_old if a.startswith('ATOM') and (a.split()[2] in [k.strip() for k in atoms])]
                pep_l = int(xyz_old[-1].split()[5])
                xyz_old = get_old_c(xyz_old, pep_l)
                sh_atoms = [stack(xyz_old[k], sh_atoms[k][3:]) for k in range(4)]       
            k = check_clashes(sh_atoms[1])
            if k == 0:
                xyz_new = make_string(ch_label, sh_atoms)
                f.writefile(pdb_name_f + pdb_name, xyz_new)
                pdb_names = [pdb_name]
    else:
        pdb_names = []
        c_names_x = [a for a in c_names if a.startswith(name)]
        print(name, c_names_x)
        for j in c_names_x:
            #print(j)
            pdb_name = '_'.join([j[:-6]] + rank + ['NC']) + '.pdb'
            #print('\t', pdb_name)
            xyz_old = f.readfile(pdb_name_f + j, 'l') 
            pep_l = int(xyz_old[-1].split()[5])
            xyz_old = get_old_c(xyz_old, pep_l)
            sh_atoms = [stack(sh_atoms[k][:3], xyz_old[k]) for k in range(4)]
            k = check_clashes(sh_atoms[1])
            if k == 0:
                xyz_new = make_string(ch_label, sh_atoms)
                f.writefile(pdb_name_f + pdb_name, xyz_new)
                pdb_names.append(pdb_name)
    return pdb_names


def align_point(p1, p2, atoms_xyz, ortha=None):
    c= np.array([0, 0, 0])
    if ortha is None:
        norm = p1 / np.linalg.norm(p1)
        ortha = np.cross(p2, norm)
    a1 = f.dihedral(np.array([p1, c, ortha, p2]))
    if a1 == a1:
        atoms_xyz = f.rotate_dihedral(a1, ortha, c, atoms_xyz)
    return atoms_xyz

def pa_align(sh, sh_atoms, pa_geom, tr):
    pa_new, pa_old = pa_geom[:, :, 0], pa_geom[:, :, 1]
    U = f.kabsch(pa_new, pa_old)[2]
    sh = np.dot(sh, U) + tr
    sh_atoms = [np.array([(np.dot(i, U) + tr).reshape(3) if i is not None else None for i in x]) for x in sh_atoms]
    return sh, sh_atoms

def align_geomsi(pg, pg_a, g1, e, hc):
    pg0 = copy.deepcopy(pg)
    pg_a0 = copy.deepcopy(pg_a)
    #"""
    print([min([f.distance(a1,a2) for a1 in pg if f.distance(a1,a2) > 0.1]) for a2 in pg])
    if (e == 0) or (e==10):
        #r = check_helix(g1, pg[2:], hc[2:])
        #if r == 1:
        c1 = g1[:3].mean(axis=0)
        c2 = pg0[3:].mean(axis=0)

        g = g1 - c1
        pg1 = pg - c2
        U = f.kabsch(pg1[3:], g[:3])[2]
        pg = np.dot(pg1, U)
        pg = pg+c1

        pg_a = [[x-c2  if x is not None else None for x in a] for a in pg_a]
        pg_a = [[np.dot(x, U) if x is not None else None for x in a] for a in pg_a]
        pg_a = [[x+c1 if x is not None else None for x in a] for a in pg_a]

    else:
        #r = check_helix(g1, pg[:4], hc[:4])
        #if r == 1:
        c1 = g1[1:].mean(axis=0)
        c2 = pg0[:3].mean(axis=0)
        g = g1 - c1
        pg1 = pg0 - c2 
        U = f.kabsch(pg1[:3], g[1:])[2]
        pg = np.dot(pg1, U)
        pg = pg+c1

        pg_a = [[x-c2  if x is not None else None for x in a] for a in pg_a]
        pg_a = [[np.dot(x, U) if x is not None else None for x in a] for a in pg_a]
        pg_a = [[x+c1 if x is not None else None for x in a] for a in pg_a]
    #"""
    return pg, pg_a

def check_against_native(sh_atoms, frags):
    sh_atoms_copy = copy.deepcopy(sh_atoms)
    CA_none_idx = [ix for ix, i in enumerate(sh_atoms[1]) if i is None]
    O_none_idx = [ix for ix, i in enumerate(sh_atoms[3]) if i is None]
    if CA_none_idx != []:
        CA_pred = np.array([a for a in sh_atoms[1] if a is not None])
    else:
        CA_pred = sh_atoms[1]
        
    if O_none_idx != []:
        O_pred = np.array([a for a in sh_atoms[3] if a is not None])
    else:
        O_pred = sh_atoms[3]
        
    tr = CA_pred.mean(axis=0)
    CA_pred -= tr
    O_pred -= tr
    d1_pred, d2_pred = f.distance(CA_pred[0], CA_pred[-1]), f.distance(O_pred[0], O_pred[-1])    
    min_rmsd = []
    for frag in frags:
        if CA_none_idx != []:
            CA_nat = np.array([a for ax, a in enumerate(frag[1]) if ax not in CA_none_idx])
        else:
            CA_nat = frag[1]
        if O_none_idx != []:
            O_nat = np.array([a for ax, a in enumerate(frag[3]) if ax not in O_none_idx])
        else:
            O_nat = frag[3]
            
        d1_nat, d2_nat = f.distance(CA_nat[0], CA_nat[-1]), f.distance(O_nat[0], O_nat[-1])
        if (abs(d1_pred - d1_nat) <= 0.8) and (abs(d2_pred - d2_nat) <= 1.0):
            rmsd1 = f.kabsch(O_nat, O_pred)[0]
            if rmsd1 < 3:
                rmsd2 = f.kabsch(CA_nat, CA_pred)
                if rmsd2[0] < 1.5:
                    min_rmsd.append([rmsd2[0], rmsd1, frag, rmsd2[2]])
    if min_rmsd == []:
        sh_atoms_new = None
    else:
        min_rmsd.sort(key=lambda x: x[0] + x[1])
        if (min_rmsd[0][0] > 0.5) and (min_rmsd[0][1] > 1.0):
            sh_atoms_new = [np.dot(x, min_rmsd[0][3])+tr for x in min_rmsd[0][2]]
        else:
            sh_atoms_new = sh_atoms_copy
    return sh_atoms_new

def calculate_rmsd(y_pred, data2, names, rank, sstr, prefix, frag_dir,  swap):
    helix_coords = get_helix_coords()
    #print(helix_coords)
    fpath = os.getcwd()
    pdb_name_f = fpath + '/' + prefix + '/binders/' 
 
    n, data, x = 0, [], 6.5
    y_pred_bb = y_pred[0][:, :6, :, :]

    pred_g1, pred_g2, pred_sh = [y_pred_bb[:, :, :, x] for x in range(3)]
    pred_atoms = [y_pred_bb[:, :, :, x] for x in range(3, y_pred_bb.shape[-1])]
    pred_g1, pred_g2 = pred_g1[:, 1:, :], pred_g2[:, :-1, :]
    #print([i['pdb'] for i in data2])
    for idx in range(len(y_pred[0])):
        #print(names[idx])
        sst_ix = sstr[idx]       
        data_i = [i for i in data2 if i['pdb'] == names[idx]][0]
        pred_g1_i, pred_g2_i, pred_sh_i = pred_g1[idx], pred_g2[idx], pred_sh[idx]
        pred_atoms_i = [x[idx] for x in pred_atoms]

        name_i = data_i['pdb']
        pa_geom, tr = [np.array(data_i[x]) for x in ['pa_xyz_or', 'tr']]
        pred_c1 = convert_dm2xyz_48f(pred_g1_i, x)
        pred_c2 = convert_dm2xyz_48f(np.flipud(pred_g2_i), x)
        if (len(pred_c1) < 2) or (len(pred_c2) < 2):
            n += 1
            continue
    
        pred_c2 = np.flipud(pred_c2)
        if (len(pred_c1) > 2) and (len(pred_c2) > 2):
            pred_c1, pred_c2 = align_geoms1(pred_c1, pred_c2)
        pred_c = np.vstack((pred_c1[:2], pred_c2[-2:]))
         
        sh2 = align_geoms(pred_sh_i, pred_c)
        rmsd = [f.rmsd(pred_c, sh2[k][1:-1]) for k in range(2)]
        if sst_ix == [0, 0]:
            s2 = [1 if check_side_of_plane(sh2[0][k:k+3], sh2[0][k+3]) >= 0 else 0 for k in range(3)]  
            s3 = [1 if check_side_of_plane(sh2[0][k:k+3], sh2[0][k+4]) >= 0 else 0 for k in range(2)]
            s4 = [1 if check_side_of_plane(sh2[0][k:k+3], sh2[0][k+5]) >= 0 else 0 for k in range(1)]          
            if s2 + s3 + s4 == [1] *6:
                k_sh = 0
            else:
                k_sh = np.argsort(rmsd)[0]
        else:
            k_sh = np.argsort(rmsd)[0]
  
        sh = sh2[k_sh] 
        
        sh_atoms = [convert_dm2xyz_48o(x, sh, xx) for xx, x in enumerate(pred_atoms_i)]
        sh, sh_atoms = pa_align(sh, sh_atoms, pa_geom, tr)
        sh_atoms = sh_atoms[:1] + [sh] + sh_atoms[1:]
        frags = f.readjson(frag_dir + 'frag_lib_' + ''.join([sst_dict[x] for x in sst_ix]) + '.json')
        if swap == 'True':
            sh_atoms = check_against_native(sh_atoms, frags)
        if sh_atoms is None:
            continue
        pdb_name = make_pdb(name_i, pdb_name_f, sh_atoms, rank, prefix)
        data += pdb_name
    return data

def get_helix_coords():
    coords= [np.array([[11.498, 57.508, 59.577],
           [11.872, 61.164, 58.606],
           [10.297, 62.253, 61.89 ],
           [12.693, 60.035, 63.837],
           [15.662, 61.532, 61.993],
           [14.461, 65.047, 62.8  ]]), np.array([[10.049, 57.508, 59.577],
           [11.462, 59.778, 58.704],
           [10.566, 61.654, 60.599],
           [11.688, 60.549, 62.93 ],
           [14.42 , 60.873, 62.342],
           [14.574, 63.681, 62.33 ]]), np.array([[12.021, 58.937, 59.577],
           [11.718, 61.852, 59.955],
           [11.37 , 61.845, 62.89 ],
           [14.018, 60.749, 63.609],
           [15.672, 62.952, 62.54 ],
           [14.675, 65.101, 64.306]]), np.array([[12.914, 59.273, 60.352],
           [12.625, 62.548, 60.405],
           [11.901, 62.687, 63.611],
           [14.665, 61.179, 64.561],
           [16.655, 63.38 , 63.141],
           [15.438, 65.965, 64.731]])]
    return coords

def calculate_rmsde(y_pred, data2, names, names_b, rank, sstr, prefix, frag_dir, e, swap, c_names = None):
    helix_coords = get_helix_coords()
    fpath = os.getcwd()
    pdb_name_f = fpath + '/' + prefix + '/binders/' 
 
    n, data, x = 0, [], 6.2
    y_pred_bb = y_pred[0][:, :6, :, :]

    pred_g1, pred_g2, pred_sh = [y_pred_bb[:, :, :, x] for x in range(3)]
    pred_atoms = [y_pred_bb[:, :, :, x] for x in range(3, y_pred_bb.shape[-1])]
    pred_g1, pred_g2 = pred_g1[:, 1:, :], pred_g2[:, :-1, :]
    #print([i['pdb'] for i in data2])
    for idx in range(len(y_pred[0])):
        
        sst_ix = sstr[idx]       
        data_i = [i for i in data2 if i['pdb'] == names[idx]][0]
        pred_g1_i, pred_g2_i, pred_sh_i = pred_g1[idx], pred_g2[idx], pred_sh[idx]
        pred_atoms_i = [x[idx] for x in pred_atoms]

        name_i = names_b[idx] # data_i['pdb']
        pa_geom, tr, p_loop = [np.array(data_i[x]) for x in ['pa_xyz_or', 'tr', 'binder_CA']]
        pred_c1 = convert_dm2xyz_48f(pred_g1_i, x)
        pred_c2 = convert_dm2xyz_48f(np.flipud(pred_g2_i), x)
        if (len(pred_c1) < 2) or (len(pred_c2) < 2):
            n += 1
            continue
    
        pred_c2 = np.flipud(pred_c2)
        if (len(pred_c1) > 2) and (len(pred_c2) > 2):
            pred_c1, pred_c2 = align_geoms1(pred_c1, pred_c2)
        pred_c = np.vstack((pred_c1[:2], pred_c2[-2:]))
         
        sh2 = align_geoms(pred_sh_i, pred_c)
        rmsd = [f.rmsd(pred_c, sh2[k][1:-1]) for k in range(2)]
        if sst_ix == [0, 0]:
            s2 = [1 if check_side_of_plane(sh2[0][k:k+3], sh2[0][k+3]) >= 0 else 0 for k in range(3)]  
            s3 = [1 if check_side_of_plane(sh2[0][k:k+3], sh2[0][k+4]) >= 0 else 0 for k in range(2)]
            s4 = [1 if check_side_of_plane(sh2[0][k:k+3], sh2[0][k+5]) >= 0 else 0 for k in range(1)]          
            if s2 + s3 + s4 == [1] *6:
                k_sh = 0
            else:
                k_sh = np.argsort(rmsd)[0]
        else:
            k_sh = np.argsort(rmsd)[0]
  
        sh = sh2[k_sh] 
        sh_atoms = [convert_dm2xyz_48o(x, sh, xx) for xx, x in enumerate(pred_atoms_i)]
        sh, sh_atoms = pa_align(sh, sh_atoms, pa_geom, tr)
        print(sst_ix)
        #print('A ', sh)
        #if sst_ix == [0, 0]:
        #   sh, sh_atoms = align_geomsi(sh, sh_atoms, p_loop, e, helix_coords[0])
        #print('B ', sh)
        #print([min([f.distance(a1,a2) for a1 in sh if f.distance(a1,a2) > 0.1]) for a2 in sh])
        sh_atoms = sh_atoms[:1] + [sh] + sh_atoms[1:]
        frags = f.readjson(frag_dir + 'frag_lib_' + ''.join([sst_dict[x] for x in sst_ix]) + '.json')
        if swap == 'True':
            sh_atoms = check_against_native(sh_atoms, frags)
        if sh_atoms is None:
            continue
        pdb_name = make_pdb(name_i, pdb_name_f, sh_atoms, rank, prefix, e, c_names)
        data += pdb_name
    return data

