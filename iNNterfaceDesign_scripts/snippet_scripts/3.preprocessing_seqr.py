import os, copy, numpy as np, sys, json, random, shutil
from modules import Class_protein as cp, Amino_acid_dictionaries as ad, functions as f, make_data_aas as mk
from pyrosetta import *
from pyrosetta.teaching import *
from pyrosetta.rosetta.protocols.relax import *
from pyrosetta.rosetta.core.pack.task import *
from pyrosetta.rosetta.core.pack.task.operation import *
from pyrosetta.rosetta.protocols.task_operations import *
from pyrosetta.rosetta.protocols.minimization_packing import *
from pyrosetta.rosetta.core.select.residue_selector import *
from pyrosetta.rosetta.core.pose import *
from pyrosetta.toolbox import *
from pyrosetta.rosetta.protocols.constraint_movers import *
from pyrosetta.rosetta.protocols.idealize import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.constraint_generator import *
from pyrosetta.rosetta.core.scoring.constraints import *

init('-ignore_unrecognized_res T -no_his_his_pairE T -no_optH F -use_input_sc T -ex1 T -ex2 T -ignore_zero_occupancy F  -detect_disulf F')
flat = lambda x: [i for j in x for i in j]
tolist = np.ndarray.tolist
np.set_printoptions(threshold=sys.maxsize)
unq = np.unique
DSSP = pyrosetta.rosetta.protocols.moves.DsspMover()

def make_folders(prefix):
    if os.path.isdir(prefix + '/binders_id') == False:
        os.mkdir(prefix + '/binders_id')
    if os.path.isdir(prefix + '/complexes') == False:
        os.mkdir(prefix + '/complexes')
    if os.path.isdir(prefix + '/scores') == False:
        os.mkdir(prefix + '/scores')
    if os.path.isdir(prefix + '/pockets') == False:
        os.mkdir(prefix + '/pockets')
        
def get_pocket_res(loop_g1_amns, all_amns):
    l_ca = [a.atoms_no_h[1].xyz for a in loop_g1_amns]
    pocket_res = [sorted([[m, f.distance(a, m.atoms_no_h[1].xyz)] for a in l_ca], key=lambda x: x[1])[0] for m in all_amns]
    l_clashes = [sorted([f.distance(l_ca[a1], l_ca[a2]) for a1 in range(len(l_ca)) if abs(a1-a2)>=2])[0] for a2 in range(len(l_ca))]
    clashes_num = len([a for a in pocket_res if a[1] <= 1])
    l_clashes = len([a for a in l_clashes if a <= 1.5])
    if (clashes_num >= 2) or (l_clashes >= 1):
        pocket_res = None
        return

    pocket_res = [n for n in pocket_res if n[1] <= 20]
    pocket_res.sort(key=lambda x: x[-1])
    pocket_res = [a[0] for a in pocket_res[:48]]
    pocket_res.sort(key=lambda x: int(x.amn_number))
    pocket_atoms = [amn.atoms_no_h for amn in pocket_res]
    pa = [[ad.amn_mapping[a[0].amn_obj.type], a[0].amn_obj.amn_number, [at.xyz for at in a]] for ax, a
          in enumerate(pocket_atoms)]

    pa_amns, amn_num, pa_xyz = [i[0] for i in pa], np.array([i[1] for i in pa]), np.array([i[2] for i in pa])
    return pocket_res, pa, pa_amns, amn_num, pa_xyz

def get_protein_amns(pdb):
    protein = cp.Protein(pdb)
    all_amns = flat([chain.amns for chain in protein.chains])
    all_amns.sort(key=lambda x: int(x.amn_number))
    clean_pose = pose_from_pdb(pdb)
    DSSP.apply(clean_pose)
    ss = clean_pose.secstruct()
    return all_amns, ss

def calc_dist(pa_xyz, loop_g1_atoms):
    dists = np.zeros((len(pa_xyz), len(loop_g1_atoms)))
    for ax1, a1 in enumerate(pa_xyz):
        for ax2, a2 in enumerate(loop_g1_atoms):
            dists[ax1, ax2] = f.distance(a1[0], a2[0].xyz)
    return dists

def calc_rmsd(loop1, loop2, l1):
    ca1 = flat([[a.atoms_no_h[1].xyz for a in loop1.chains[x].amns] for x in range(len(loop1.chains))])
    ca2 = [a.atoms_no_h[1].xyz for a in loop2.chains[0].amns]
    ca1_amn = ''.join(flat([[a.type1l for a in loop1.chains[x].amns] for x in range(len(loop1.chains))]))
    rmsd4 = [f.rmsd(ca2, ca1[k:k +l1 ]) for k in range(len(ca1)-l1 +1)]
    k = np.argmin(rmsd4)
    seq1 = ca1_amn[k:k+l1]

    ca1 = ca1[::-1]
    ca1_amn = ''.join([a for a in ca1_amn][::-1])
    rmsd4r = [f.rmsd(ca2, ca1[k:k + l1]) for k in range(len(ca1)-l1+1)]
    kr = np.argmin(rmsd4r)
    seq2 = ca1_amn[kr:kr+l1]
    return [int(k), rmsd4[k], int(kr), rmsd4r[kr], seq1, seq2]


def get_frags_end(dir, cpxs, pdb, prefix):
    make_folders(prefix)
    id_mover = IdealizeMover()
    data_all = []
    pdb = pdb + 't.pdb'
    all_amns, ss = get_protein_amns(pdb)
   
    for cx, cpx in enumerate(cpxs[:]):
        loop_g1 = pose_from_pdb(dir + cpx)
        """
        loop_g1.delete_residue_range_slow(13, loop_g1.total_residue())

        if len(loop_g1.sequence()) % 3 != 0 :
            print('Heavy atoms are missed: ', cpx)
            continue
        for i in range(1,loop_g1.total_residue()+1):
            mutate_residue(loop_g1, i, 'G')   
        
        id_mover.apply(loop_g1)
        """
        loop_g1.dump_pdb(prefix + '/binders_id/' + cpx[:-6] + '_id.pdb')
        loop_g1 = cp.Protein(prefix + '/binders_id/' + cpx[:-6] + '_id.pdb')
        loop_g1_amns = sorted(flat([ch.amns for ch in loop_g1.chains]),key=lambda x: int(x.amn_number))
        
        loop_g1_atoms = [amn.atoms_no_h for amn in loop_g1_amns]
        loop_g1_atoms_xyz = [[a.xyz for a in amn] for amn in loop_g1_atoms]
        p_frags = list(range(0, len(loop_g1_amns)))[:-5]
         
        p_frags = [[x, x+6] for x in p_frags]
        pocket_res_all = []
        for px, p_f in enumerate(p_frags):
            pocket_info = get_pocket_res(loop_g1_amns[p_f[0]:p_f[1]], all_amns)
            if pocket_info is None:
                continue
            pocket_res, pa, pa_amns, amn_num, pa_xyz = pocket_info
            dists = calc_dist(pa_xyz, loop_g1_atoms)
            ss_p = [ss[int(a.amn_number) - 1] for a in pocket_res]
 
            data_cpx = {'binder_struct': loop_g1_atoms_xyz[p_f[0]:p_f[1]], 'dists': dists.tolist(), 'pdb_binder': cpx, 'homo': 0, 'secondary_p': ss_p, 'pocket_struct': pa, 'p_f': px}
            data_all.append(data_cpx)
            pocket_res_all += [a for a in pocket_res if a not in pocket_res_all]
        pocket_res_all.sort(key=lambda x: int(x.amn_number))
        pocket_amns_pdb = ''.join([amn.amn_pdb for amn in pocket_res_all])
        f.writefile(prefix + '/pockets/' + cpx[:-4] + 'pocket.pdb', pocket_amns_pdb)
        
    return data_all

targ = '6vy4'
dir  = '6vy4_HNCnew/binders/'
#dir  = '/home/raulia/amino_acid_bb/6vy4_r/binders_id/'
prefix = targ + '_Hr9'
cpx = list(filter(lambda x: x.endswith('_r.pdb'), os.listdir(dir)))
data_all = []
for x in ['6vy4']:
     cpx_x = [a for a in cpx if a.startswith(x)]
     data_all += get_frags_end(dir, cpx_x, x, prefix)
print(len(data_all))
data= mk.get_data(data_all)
with open(prefix + '/' + prefix + '_data_aas.json', 'w') as outfile:
    json.dump(data, outfile)

