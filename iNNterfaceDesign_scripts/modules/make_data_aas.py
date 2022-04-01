import numpy as np, json, random, copy, random
from modules import functions as f 

BQ = lambda x: [[x, 0, 0], [0, x, 0], [0, 0, x], [-x, 0, 0], [0, -x, 0], [0, 0, -x]]
c = np.array([0, 0, 0])
st = {'H': 0, 'E': 1, 'L': 2}


def closest(l, k):
    l.sort(key=lambda x: x[k])
    return l[0]


def make_amn_seq(amns_p, d):
    amns = f.one_hot_amns(amns_p)
    amns = np.vstack((amns, np.zeros((d, 20)))) if ((d > 0) and (len(amns_p) > 6)) else amns
    amns = np.hstack((np.zeros(len(amns)).reshape(-1, 1), amns))
    return amns


def input_1(xyz_p, amns_p, la_xyz, d, sec):
    amns = np.array(amns_p)
    amns = np.hstack((amns, np.zeros((d)))) if d > 0 else amns
    sec = np.array([st[s] + 1 for s in sec])
    sec = np.hstack((sec, np.zeros((d)))) if d > 0 else sec

    ats = []
    for ax, at in enumerate(xyz_p):
        dists = np.array([[f.distance(a1_xyz, a2_xyz) for a1_xyz in la_xyz] for a2_xyz in at])
        if d > 0:
            dists = np.vstack((dists, np.zeros((d, len(BQ(1))))))
        ats.append(dists)
    """
    sec_v = np.zeros((len(dists), 6))
    for sx, s in enumerate(sec):
        v = st[s]
        sec_v[sx, v*2] = 1
        sec_v[sx, v*2+1] = 1
    """
    ats = np.dstack(ats)  # + [sec_v])
    return [ats, amns, sec]


def input_2(core, amn_num, d):
    mask = np.array([0 if int(i) in core else 1 for idx, i in enumerate(amn_num)])
    mask = np.hstack([mask.reshape(-1, 1)] * 6)
    mask = np.vstack((mask, np.zeros((d, 6)))) if d > 0 else mask
    return [np.expand_dims(mask, axis=2)]


def input_3(xyz_l):
    dists = [np.array([[f.distance(a1, a2) for a1 in xyz_l[1]] for a2 in xyz_l[k]]) for k in range(len(xyz_l))]
    dists = np.dstack(dists)
    return [dists]

def select_k(pa_xyz, pa_amns, amn_num, set_ss, idx):
    idx.sort()    
    set_ss = [set_ss[k] for k in idx]
    pa_xyz = [np.array([pa[x] for x in idx]) for pa in pa_xyz]
    pa_amns = np.array([pa_amns[x] for x in idx])
    amn_num = [amn_num[x] for x in idx]
    return pa_xyz, pa_amns, amn_num, set_ss
 
def get_data(batch):
    inputs = []
    k1 = 48
    scan_all = []
    for cx, cpx in enumerate(batch[:]): 
        pa, la, homo, name = cpx['pocket_struct'], cpx['binder_struct'], cpx['homo'], cpx['pdb_binder']
        pa_amns, amn_num, pa_xyz = [i[0] for i in pa], np.array([i[1] for i in pa]), [i[2] for i in pa]
        pa_xyz = [np.array([pa_xyz[i][k] for i in range(len(pa_xyz))]) for k in [0, 3]]
        la_xyz = [np.array([la[i][k] for i in range(len(la))]) for k in [0, 3]]
        la_xyz_ca = la_xyz[0]
        set_ss, min_dists = cpx['secondary_p'], np.array(cpx['dists']) 
        min_dists = min_dists.min(axis=1)
        idx = np.argsort(min_dists)[:k1]

        pa_xyz_k1, pa_amns_k1, amn_num_k1, set_ss_k1 = select_k(pa_xyz, pa_amns, amn_num, set_ss, idx)
        d_k1 = k1 - (len(pa_xyz_k1[0])) if len(pa_xyz_k1[0]) < k1 else 0

        inputs_cpx = input_1(pa_xyz_k1, pa_amns_k1, la_xyz_ca, d_k1, set_ss_k1)
        inputs_cpx += input_3(la_xyz)
        inputs_cpx += [int(homo), name]        
        if 'p_f' in cpx:
            inputs_cpx += [cpx['p_f']]
        inputs.append(inputs_cpx)
    if inputs != []:
        inputs = [np.array([inputs[i][j] for i in range(len(inputs))]) for j in range(len(inputs[0]))]
    for i in range(len(inputs)):
        print(inputs[i].shape)
    print(len(inputs[-1]), len(np.unique(inputs[-1])))
    inputs = [i.tolist() for i in inputs]
    return inputs


