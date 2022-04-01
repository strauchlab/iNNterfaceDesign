import numpy as np, json,  random, copy, random
from modules import functions as f


BQ = lambda x: [[x, 0, 0], [0, x, 0], [0, 0, x], [-x, 0, 0], [0, -x, 0], [0, 0, -x]]
c = np.array([0, 0, 0])
st = {'H': 0, 'E': 1, 'L': 2}

def closest(l, k):
    l.sort(key=lambda x: x[k])
    return l[0]

def input_1(xyz_p, d, x, sec, amns_p = None):
    if amns_p is not None:
        amns_p = np.array(amns_p)
        amns_p = np.hstack((amns_p, np.zeros((d)))) if d > 0 else amns_p
    sec = np.array([st[s]+1 for s in sec])
    sec = np.hstack((sec, np.zeros((d)))) if d > 0 else sec

    ats = []
    for ax, at in enumerate(xyz_p):
        dists = np.array([[f.distance(a1_xyz, a2_xyz) for a1_xyz in BQ(x)] for a2_xyz in at])
        if d> 0:
            dists = np.vstack((dists, np.zeros((d, len(BQ(1))))))
        ats.append(dists)
    ats = np.dstack(ats)
    return [ats, sec, amns_p]

def input_2(core, amn_num, d,  close):
    mask1 = []
    if close is not None:
        for idx, i in enumerate(amn_num):
            if (int(i) in close) and (int(i) not in core):
                mask1.append(1)
            elif (int(i) not in core):
                mask1.append(0.6)
            else:
                mask1.append(0.2)
    elif close is None:
        mask1 = [0.2 if int(i) in core else 0.6 for idx, i in enumerate(amn_num)]
    mask = np.hstack((np.array(mask1), np.zeros((d)))) if d > 0 else np.array(mask1)
    return [mask]

def shape_ar(xyz_l):
    dists = [np.array([[f.distance(a1, a2) for a1 in xyz_l[1]] for a2 in xyz_l[1]])]
    for k in [0, 2, 3]:
        dists_k = np.array([[f.distance(a1, a2) if ax1 in np.argsort([abs(x-ax2) for x in range(6)])[:4] else 0 for ax1, a1 in enumerate(xyz_l[1])] for ax2, a2 in enumerate(xyz_l[k])])
        dists.append(dists_k)
    dists = np.dstack(dists)
    return dists


def input_3(xyz_l, x, e):
    BQx = BQ(x)
    bn = len(BQx) 
    na = 6

    arrs1, arrs2 = [], []
    shape = shape_ar(xyz_l)
    #shape = shape[:, :, [0, -1]]        
    at1 = BQx + xyz_l[1].tolist()
    at2 = BQx + xyz_l[1][::-1].tolist()
    for idx in range(len(at1[bn:])):
        if idx != 0:
            res1 = at1[:bn-min(idx, na)] + at1[max(bn, bn-na+idx) : bn+idx+1]
            res2 = at2[:bn-min(idx, na)] + at2[max(bn, bn-na+idx) : bn+idx+1]
            #res2 = at2[:bn-idx] + at2[bn : bn+idx+1]
        else:
            res1 = at1[:bn+1]
            res2 = at2[:bn+1]

        arr1 = np.array([[round(f.distance(i, res1[a]), bn + 1) for a in range(bn + 1)] for i in res1])
        arr1 = arr1[-1][:-1]
        arrs1.append(arr1)
        arr2 = np.array([[round(f.distance(i, res2[a]), bn + 1) for a in range(bn + 1)] for i in res2])
        arr2 = arr2[-1][:-1]
        arrs2.append(arr2)
    arrs1 = np.array(arrs1)
    arrs2 = np.array(arrs2)
    
    if e == 0:
        shape = np.hstack((np.zeros((4, 2, 4)), shape))
        ats = np.dstack((arrs1, arrs2, shape))
    else:
        shape = np.hstack((shape, np.zeros((4, 2, 4))))
        ats = np.dstack((arrs1, arrs2, shape))
    return [ats]

def select_k1(pa_xyz, amn_num_l, tr, ld, ldd, la_xyz):
    pa_xyz = [pa_xyz[i] - tr for i in range(len(pa_xyz))]
    if la_xyz is not None:
        la_xyz = [la_xyz[i] - tr for i in range(len(la_xyz))]

    pa_xyz_or = copy.deepcopy(pa_xyz[0])
    ld = [a-tr if a is not None else None for a in ld]
    ldd = [a-tr if a is not None else None for a in ldd]

    tr1 = pa_xyz[0][amn_num_l[0]]
    if not ((abs(tr1[1]) < 0.2) and (abs(tr1[2]) < 0.2)):
        pa_xyz = [align_point(tr1, np.array([1, 0, 0]), pa_xyz[i]) for i in range(len(pa_xyz))]
        if la_xyz is not None:
            la_xyz = [align_point(tr1, np.array([1, 0, 0]), la_xyz[i]) for i in range(len(la_xyz))]
        ld = [align_point(tr1, np.array([1, 0, 0]), a) if a is not None else None for a in ld]
        ldd = [align_point(tr1, np.array([1, 0, 0]), a) if a is not None else None for a in ldd]
        
    else:
        if tr1[0] < 0:
            pa_xyz = [align_point(tr1, np.array([1, 1, 0]), pa_xyz[i]) for i in range(len(pa_xyz))]
            pa_xyz = [align_point(np.array([1, 1, 0]), np.array([1, 0, 0]), pa_xyz[i]) for i in range(len(pa_xyz))]
            ld = [align_point(tr1, np.array([1, 1, 0]), a) if a is not None else None for a in ld]           
            ld = [align_point(np.array([1, 1, 0]), np.array([1, 0, 0]), a) if a is not None else None for a in ld]
            ldd = [align_point(tr1, np.array([1, 1, 0]), a) if a is not None else None for a in ldd]
            ldd = [align_point(np.array([1, 1, 0]), np.array([1, 0, 0]), a) if a is not None else None for a in ldd]
            
            if la_xyz is not None:
                la_xyz = [align_point(tr1, np.array([1, 1, 0]), la_xyz[i]) for i in range(len(la_xyz))]
                la_xyz = [align_point(np.array([1, 1, 0]), np.array([1, 0, 0]), la_xyz[i]) for i in range(len(la_xyz))]

    tr2 = pa_xyz[0][amn_num_l[-1]]
    if not ((abs(tr2[1]) < 0.3) and (abs(tr2[2]) < 0.3)):
        pa_xyz = [align_point(tr2, np.array([-1, 0.3, 0]), pa_xyz[i], ortha=np.array([1, 0, 0])) for i in
                  range(len(pa_xyz))]
        ld = [align_point(tr2, np.array([-1, 0.3, 0]), a) if a is not None else None for a in ld]
        ldd = [align_point(tr2, np.array([-1, 0.3, 0]), a) if a is not None else None for a in ldd]
        if la_xyz is not None:
            la_xyz = [align_point(tr2, np.array([-1, 0.3, 0]), la_xyz[i], ortha=np.array([1, 0, 0])) for i in range(len(la_xyz))]

    pa_xyz_or = np.dstack((pa_xyz[0], pa_xyz_or))
    ld = [np.unique([np.argsort([f.distance(j, x) for x in BQ(10)])[:2] for j in a]).tolist()  if a is not None else None for a in ld]
    ldd = [np.unique([np.argsort([f.distance(j, x) for x in BQ(10)])[:2] for j in a]).tolist() if a is not None else None for a in ldd]
    #print(ldd)
    return pa_xyz, pa_xyz_or, tr, ld, ldd, la_xyz


def select_k2(pa_xyz, amn_num, k, set1_ss, en, set1, pa_amns):
    idx_s = set1[:k]
    idx = np.array([idx for idx, i in enumerate(amn_num) if int(i) in idx_s])
    set_ss = [set1_ss[k] for k in idx]
    pa_xyz = [np.array([pa[x] for x in idx]) for pa in pa_xyz]
    en = [en[x] for x in idx]
    pa_amns = [pa_amns[x] for x in idx]
    return pa_xyz, set_ss, en, pa_amns


def align_point(p1, p2, atoms_xyz, ortha=None):
    c = np.array([0, 0, 0])
    if ortha is None:
        norm = p1 / np.linalg.norm(p1)
        ortha = np.cross(p2, norm)
    a1 = f.dihedral(np.array([p1, c, ortha, p2]))
    if a1 == a1:
        atoms_xyz = f.rotate_dihedral(a1, ortha, c, atoms_xyz)
    return atoms_xyz


def make_data(batch):
    k1, x = 48, 6
    inputs, inputs2 = [], []
    for cx, cpx in enumerate(batch[:]):
        la_xyz = None
        pa = cpx['pocket_struct']
        p_ss = cpx['secondary_p']
        core, cr, p_res, idx10, ld, ldd = cpx['core'], cpx['central_resi'], cpx['set1'], cpx['ppi'], cpx['ld'], cpx['ldd']
        if 'binder_xyz' in cpx:
            la_end, la_xyz = cpx['binder_end'], cpx['binder_xyz']
            la_xyz = [np.array([la_xyz[i][k] for i in range(len(la_xyz))]) for k in range(4)]
            la_xyz_CA = la_xyz[1]

        pa_amns, amn_num, pa_xyz = [i[0] for i in pa], np.array([i[1] for i in pa]), np.array([i[2] for i in pa])
        pa_xyz = [np.array([pa_xyz[i][k] for i in range(len(pa_xyz))]) for k in [1, 3]]

        idx = [adx for adx, a in enumerate(amn_num) if a in p_res[:k1]]
        p_ss = [p_ss[k] for k in idx]
        pa_xyz = [np.array([pa[x] for x in idx]) for pa in pa_xyz]
        pa_amns = np.array([pa_amns[x] for x in idx])
        amn_num = [amn_num[x] for x in idx]

        cr_idx = [idx for idx, i in enumerate(amn_num) if int(i) == cr]
        amn_num_l = [idx for idx, i in enumerate(amn_num) if int(i) in list(range(cr - 4, cr + 5))]
        if (amn_num_l == []) or (cr_idx == []):
            continue

        tr = pa_xyz[0][cr_idx][0]
        pa_xyz_k1, pa_xyz_or, tr, ld, ldd, la_xyz = select_k1(pa_xyz, amn_num_l, tr, ld, ldd, la_xyz)
        
        d_k1 = k1 - (len(pa_xyz_k1[0])) if len(pa_xyz_k1[0]) < k1 else 0
        inputs_cpx = input_1(pa_xyz_k1, d_k1, x, p_ss, pa_amns)
        inputs_cpx += input_2(core, amn_num, d_k1, idx10)
        inputs_cpx += [cpx['pdb']]
        if la_xyz is not None:
            la_end = cpx['binder_end']
            inputs_cpx += [cpx['binder_name']]
            inputs_cpx += input_3(la_xyz, x, la_end)
            inputs_cpx += [la_end]
        else:
            inputs_cpx += [cpx['pdb']]
        inputs.append(inputs_cpx)

        cpx.update({'pa_xyz_or': [i.tolist() for i in pa_xyz_or], 'tr': tr.tolist(), 'ld': ld, 'ldd': ldd})
        if la_xyz is not None:
            cpx.update({'binder_CA': la_xyz_CA.tolist()})
        inputs2.append(cpx)
    #print(inputs)
    if inputs != []:
        inp = [np.array([inputs[i][j] for i in range(len(inputs))]) for j in range(len(inputs[0]))]
        for i in range(len(inp)):
            print(inp[i].shape)

        inputs = {'p_xyz1': inp[0], 'sec1': inp[1], 'p_amn1': inp[2], 'mask': inp[3], 'name': inp[4]}
        if la_xyz is not None:
            inputs.update({'binder_name': inp[5], 'binder_xyz': inp[6], 'b_end': inp[7]})
        inputs = {a: inputs[a].tolist() for a in inputs}
    return inputs, inputs2


