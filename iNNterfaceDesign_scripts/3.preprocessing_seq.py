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
from pyrosetta.rosetta.core.io.pose_from_sfr import PoseFromSFRBuilder

init('-ignore_unrecognized_res T -no_his_his_pairE T -no_optH F -use_input_sc T -ex1 T -ex2 T -ignore_zero_occupancy F  -detect_disulf F')
flat = lambda x: [i for j in x for i in j]
tolist = np.ndarray.tolist
np.set_printoptions(threshold=sys.maxsize)
unq = np.unique
sw1 = SwitchResidueTypeSetMover("centroid")
sw2 = SwitchResidueTypeSetMover("fa_standard")

def sc_cen_constr():
    #scorefxn = ScoreFunctionFactory.create_score_function("ref2015_cart.wts")
    scorefxn = ScoreFunctionFactory.create_score_function("cen_std")
    scorefxn.set_weight(coordinate_constraint, 1.0)
    return scorefxn


def make_folders(prefix):
    if os.path.isdir(prefix + '/binders_id') == False:
        os.mkdir(prefix + '/binders_id')
    if os.path.isdir(prefix + '/binders_rel') == False:
        os.mkdir(prefix + '/binders_rel')
    if os.path.isdir(prefix + '/complexes') == False:
        os.mkdir(prefix + '/complexes')
    if os.path.isdir(prefix + '/scores') == False:
        os.mkdir(prefix + '/scores')
        
def get_pocket_res(loop_g1_amns, all_amns):
    l_ca = [a.atoms_no_h[1].xyz for a in loop_g1_amns]
    core_amns = f.readjson(prefix + '/' + prefix + '_2b.json')[0]['core']
    pocket_res = [sorted([[m, f.distance(a, m.atoms_no_h[1].xyz)] for a in l_ca], key=lambda x: x[1])[0] for m in all_amns]

    clashes_num = len([a for a in pocket_res if a[1] <= 2.5])
    if (clashes_num >= 2):
        pocket_res = None
        return
    cthr = 3.0 #if len(l_ca) == 6 else 5
    clashes = [sorted([f.distance(a, m.atoms_no_h[1].xyz) for m in all_amns if int(m.amn_number) in core_amns])[0] for ax, a in enumerate(l_ca) if ax in [0, 1, len(l_ca)-2, len(l_ca)-1]]
    if (len([a for a in clashes[:2] if a < cthr]) ==2) or (len([a for a in clashes[2:] if a < cthr]) ==2): 
        pocket_res = None
        return

    pocket_res = [n for n in pocket_res if n[1] <= 20]
    pocket_res.sort(key=lambda x: x[1])
    pocket_res = [a[0] for a in pocket_res[:48]]
    pocket_res.sort(key=lambda x: int(x.amn_number))
    pocket_atoms = [amn.atoms_no_h for amn in pocket_res]
    pa = [[ad.amn_mapping[a[0].amn_obj.type], a[0].amn_obj.amn_number, [at.xyz for at in a]] for ax, a
          in enumerate(pocket_atoms)]

    pa_amns, amn_num, pa_xyz = [i[0] for i in pa], np.array([i[1] for i in pa]), np.array([i[2] for i in pa])
    return pocket_res, pa, pa_amns, amn_num, pa_xyz

def get_protein_amns(keywords):
    protein = cp.Protein(keywords['pdb'] + '.pdb')
    all_amns = flat([chain.amns for chain in protein.chains])
    all_amns.sort(key=lambda x: int(x.amn_number))
    return all_amns

def calc_dist(pa_xyz, loop_g1_atoms):
    dists = np.zeros((len(pa_xyz), len(loop_g1_atoms)))
    for ax1, a1 in enumerate(pa_xyz):
        for ax2, a2 in enumerate(loop_g1_atoms):
            dists[ax1, ax2] = f.distance(a1[0], a2[0].xyz)
    return dists

def centroid_mover(scorefxn, loop_amn):
    loop_amn = np.array(loop_amn) + 1
    restrict1 = RestrictToRepacking()
    tf = TaskFactory()
    tf.push_back(InitializeFromCommandline())
    tf.push_back(restrict1)

    mm = MoveMap()
    for i in loop_amn:
        mm.set_bb(i, True)
      
    relax = CentroidRelax()
    relax.set_score_function(scorefxn)
    relax.set_ramp_rama(True)
    #relax.constrain_relax_to_native_coords(True)
    relax.min_type("lbfgs_armijo")
    relax.max_iter(200)
    relax.set_movemap(mm)
    relax.set_task_factory(tf)
    return relax


def get_frags(cpxs, prefix, data_info, keywords):
    print(cpxs)
    prefix = keywords['prefix']
    make_folders(prefix)
    id_mover = IdealizeMover()
    data_all = []
    all_amns = get_protein_amns(keywords)
    #loop_nat = cp.Protein('3ztj_HG.pdb')

    scorefxn = sc_cen_constr()
    for cx, cpx in enumerate(cpxs[:]):
        if (os.path.exists(prefix + '/binders/' + cpx) == True):
            name = '_'.join(cpx.split('_')[:-2])
            data_f = [a for a in data_info if a['pdb'] == name][0]
            ss = data_f['secondary_p_all']
            data_cpx = copy.deepcopy(data_f)
            pocket = pose_from_pdb(prefix + '/pockets/' + name + 'pocket.pdb')
            loop_g1 = pose_from_pdb(prefix + '/binders/' + cpx)
            if len(loop_g1.sequence()) != 6:
                print('Heavy atoms are missed: ', cpx)
                continue

            id_mover.apply(loop_g1)
            """
            loop_amn = list(range(1, loop_g1.total_residue()+1))
            cen_relax = centroid_mover(scorefxn, loop_amn)
            sw1.apply(loop_g1)
            cen_relax.apply(loop_g1)
            sw2.apply(loop_g1)
            """
            loop_g1.dump_pdb(prefix + '/binders_id/' + cpx[:-4] + '_id.pdb')        
            append_pose_to_pose(loop_g1, pocket)
            loop_g1.dump_pdb(prefix + '/complexes/' + cpx[:-4] + '.pdb')
            loop_g1 = cp.Protein(prefix + '/binders_id/' + cpx[:-4] + '_id.pdb')
            loop_g1_amns = sorted(flat([ch.amns for ch in loop_g1.chains]),key=lambda x: int(x.amn_number))
            #if len(loop_g1_amns) <= 6:
            #    rmsd = calc_rmsd(loop_nat, loop_g1, len(loop_g1_amns))
            #    if min(rmsd[1], rmsd[3]) > 6:
            #        continue
 
            loop_g1_atoms = [amn.atoms_no_h for amn in loop_g1_amns]
            loop_g1_atoms_xyz = [[a.xyz for a in amn] for amn in loop_g1_atoms]
            pocket_info = get_pocket_res(loop_g1_amns, all_amns)
            if pocket_info is None:
                if os.path.exists(prefix + '/binders_id/' + cpx[:-4] + '_id.pdb'):
                    os.remove(prefix + '/binders_id/' + cpx[:-4] + '_id.pdb')
                if os.path.exists(prefix + '/complexes/' + cpx[:-4] + '.pdb'):
                    os.remove(prefix + '/complexes/' + cpx[:-4] + '.pdb')
                continue
            
            pocket_res, pa, pa_amns, amn_num, pa_xyz = pocket_info

            dists = calc_dist(pa_xyz, loop_g1_atoms)
            ss = [ss[int(a.amn_number) - 1] for a in pocket_res]
            data_cpx.update({'binder_struct': loop_g1_atoms_xyz, 'dists': dists.tolist(), 'pdb_binder': cpx, 'homo': keywords['interf_type'], 'secondary_p': ss, 'pocket_struct': pa, 'p_f': -1})
            data_all.append(data_cpx)
            
    return data_all 

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


def get_frags_end(cpxs, prefix, data_info, keywords):
    all_amns = get_protein_amns(keywords)
    id_mover = IdealizeMover()
    data_all = []
    print([a['pdb'] for a in data_info])
    print([a['binder_name'] for a in data_info])
    #loop_nat = cp.Protein('3ztj_HG.pdb')
    scorefxn = sc_cen_constr()

    for cx, cpx in enumerate(cpxs[:]):
        data_cpx_all = []
        namei = cpx[len(prefix)+1:].split('_')[0]
        namei = '_'.join([prefix, namei])
 
        data_fs = [a for a in data_info if a['binder_name'] in cpx]
        ss = data_fs[0]['secondary_p_all']

        loop_g1 = pose_from_pdb(prefix + '/binders/' + cpx)
        if len(loop_g1.sequence()) % 3 != 0 :
            print('Heavy atoms are missed: ', cpx)
            continue
            
        id_mover.apply(loop_g1)
        """
        loop_amn = list(range(1, loop_g1.total_residue()+1))
        cen_relax = centroid_mover(scorefxn, loop_amn)
        sw1.apply(loop_g1)
        cen_relax.apply(loop_g1)
        sw2.apply(loop_g1)
        """
        loop_g1.dump_pdb(prefix + '/binders_id/' + cpx[:-4] + '_id.pdb')
        pocket = pose_from_pdb(prefix + '/pockets/' + namei + 'pocket.pdb')
        
        append_pose_to_pose(loop_g1, pocket)
        loop_g1.dump_pdb(prefix + '/complexes/' + cpx[:-4] + '.pdb')
           
        loop_g1 = cp.Protein(prefix + '/binders_id/' + cpx[:-4] + '_id.pdb')
        loop_g1_amns = sorted(flat([ch.amns for ch in loop_g1.chains]),key=lambda x: int(x.amn_number))
        #if len(loop_g1_amns) <= 12:
        #    rmsd = calc_rmsd(loop_nat, loop_g1, len(loop_g1_amns))
        #    if min(rmsd[1], rmsd[3]) > 7:
        #        continue
          

        loop_g1_atoms = [amn.atoms_no_h for amn in loop_g1_amns]
        loop_g1_atoms_xyz = [[a.xyz for a in amn] for amn in loop_g1_atoms]
        p_frags = list(range(0, len(loop_g1_amns)))[:-5]
         
        p_frags = [[x, x+6] for x in p_frags]
        thr_dist = 0
        pocket_res_all = []
        for px, p_f in enumerate(p_frags):
            pocket_info = get_pocket_res(loop_g1_amns[p_f[0]:p_f[1]], all_amns)
            if pocket_info is None:
                if os.path.exists(prefix + '/binders_id/' + cpx[:-4] + '_id.pdb'):
                    os.remove(prefix + '/binders_id/' + cpx[:-4] + '_id.pdb')
                if os.path.exists(prefix + '/complexes/' + cpx[:-4] + '.pdb'):
                    os.remove(prefix + '/complexes/' + cpx[:-4] + '.pdb')
                continue

            pocket_res, pa, pa_amns, amn_num, pa_xyz = pocket_info
            dists = calc_dist(pa_xyz, loop_g1_atoms[p_f[0]:p_f[1]])
            l_dists = dists.min(axis=0)
            if (px == 0) and (l_dists[0] >= 10):
                thr_dist += 1
            if (px == len(p_frags)-1) and (l_dists[-1] >= 10):
                thr_dist += 1
            ss_p = [ss[int(a.amn_number) - 1] for a in pocket_res]
 
            data_cpx = {'binder_struct': loop_g1_atoms_xyz[p_f[0]:p_f[1]], 'dists': dists.tolist(), 'pdb_binder': cpx, 'homo': keywords['interf_type'], 'secondary_p': ss_p, 'pocket_struct': pa, 'p_f': px}
            data_cpx_all.append(data_cpx)
            pocket_res_all += [a for a in pocket_res if a not in pocket_res_all]
        #print(thr_dist)
        #if thr_dist == 2:
        data_all += data_cpx_all
        pocket_res_all.sort(key=lambda x: int(x.amn_number))
        pocket_amns_pdb = ''.join([amn.amn_pdb for amn in pocket_res_all])
        f.writefile(prefix + '/pockets/' + cpx[:-4] + 'pocket.pdb', pocket_amns_pdb)
        #else:
        #    os.remove(prefix + '/binders_id/' + cpx[:-4] + '_id.pdb')
        #    os.remove(prefix + '/complexes/' + cpx[:-4] + '.pdb')
        #    os.remove(prefix + '/binders/' + cpx)
    return data_all

input_file = sys.argv[1]
keywords = f.keywords(input_file)
prefix = keywords['prefix']
data_info = f.readjson(prefix + '/' + prefix  +  '_2b.json')
cpxs = f.readjson(prefix + '/' + prefix  + '_names.json')
if 'add_residues' not in keywords:
    data_all = get_frags(cpxs, prefix, data_info, keywords)
else:
    data_all = get_frags_end(cpxs, prefix, data_info, keywords)
#print(len(data_all))
data= mk.get_data(data_all)
with open(prefix + '/' + prefix + '_data_aas.json', 'w') as outfile:
    json.dump(data, outfile)

