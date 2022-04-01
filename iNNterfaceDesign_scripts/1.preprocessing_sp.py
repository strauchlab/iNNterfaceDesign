from modules import functions as f, geometry_functions as fm, make_input, Class_protein as cp
from modules import Amino_acid_dictionaries as ad, make_data_aas as mk
import os, numpy as np, sys
from rosetta.utility import vector1_string
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

init('-ignore_unrecognized_res T -no_his_his_pairE T -no_optH F -use_input_sc T -ex1 T -ex2 T')

flat = lambda x: [i for j in x for i in j]
tolist = np.ndarray.tolist
DSSP = pyrosetta.rosetta.protocols.moves.DsspMover()
rev_ad = {ad.amn_mapping1letter[i]:i for i in ad.amn_mapping1letter}
id_mover = IdealizeMover()

def make_folders(prefix):
    if os.path.isdir(prefix) == False:
        os.mkdir(prefix)
    if os.path.isdir(prefix + '/structures') == False:
        os.mkdir(prefix + '/structures')
    if os.path.isdir(prefix + '/binders_rel') == False:
        os.mkdir(prefix + '/binders_rel')
    if os.path.isdir(prefix + '/scores') == False:
        os.mkdir(prefix + '/scores')



def get_protein(pdb):
    protein = cp.Protein(pdb + '.pdb')
    all_amn = flat([chain.amns for chain in protein.chains])
    clean_pose = pose_from_pdb(pdb + '.pdb')
    DSSP.apply(clean_pose)
    ss = clean_pose.secstruct()
    return protein, all_amn, ss

def get_binder_res_num(keywords):
    res1_int, res_int = [], []
    if ('res1' in keywords) and (keywords['res1'] != ''):
        res1 = flat(f.get_explicit_numbers(keywords['res1']))
        res1_int = [list(range(a,a+6)) for a in res1]
    if ('res_interval' in keywords) and (keywords['res_interval'] != ''):
        res_int = f.get_explicit_numbers(keywords['res_interval'])
    return res1_int, res_int

def get_pocket_res(binder_res, all_amn, keywords):
    pocket_res = np.unique(flat([[r for r in a.close if a.close != {}] for a in binder_res]))
    pocket_res = [a for a in all_amn if a.amn_name in pocket_res]
    if ('p_chains' in keywords):
        p_chains = [a.strip() for a in keywords['p_chains'].split(',')]
        pocket_res = [a for a in pocket_res if a.chain_obj.label in p_chains]
    return pocket_res

def get_pocket_res_fragm(fragm, j, pocket_res, keywords):
    pocket_res_n = np.unique(flat([[r for r in a.close if a.close != {}] for a in j]))
    pocket_res_fragm = [a for a in pocket_res if (a.amn_name in pocket_res_n) and (a not in j)]
    pocket_res_fragm = [a for a in pocket_res_fragm if
                  len([x for x in range(int(a.amn_number) - 2, int(a.amn_number) + 3) if x in fragm]) == 0]
    if ('use_prot_lig' not in keywords) or (keywords['use_prot_lig'] == 'False'):
        pocket_res_fragm = [a for a in pocket_res_fragm if j[0].chain_obj.label != a.chain_obj.label]
    pocket_res_ca = [amn.atoms_no_h[1] for amn in pocket_res_fragm]
    pocket_res_fragm = [sorted([[m.amn_obj, f.distance(a.atoms_no_h[1].xyz, m.xyz)] for a in j], key=lambda x: x[1])[0] for m in
                  pocket_res_ca]
    pocket_res_fragm = [n for n in pocket_res_fragm if n[1] <= 20]
    if len(pocket_res_fragm) < 24:
        pocket_res_fragm = None
    else:
        pocket_res_fragm.sort(key=lambda x: x[-1])
        pocket_res_fragm = pocket_res_fragm[:48]
        pocket_res_fragm = [n[0] for n in pocket_res_fragm]
        pocket_res_fragm.sort(key=lambda x: int(x.amn_number))
    return pocket_res_fragm

def make_all_gly(binder_pose, keywords):
    for i in range(1, 7):
        mutate_residue(binder_pose, i, 'G')
    if ('idealize' not in keywords) or (keywords['idealize'] == 'True'):
        id_mover.apply(binder_pose)
    return binder_pose

def calc_dist(pocket_atoms, binder_atoms):
    dists = np.zeros((len(pocket_atoms), len(binder_atoms)))
    for ax1, a1 in enumerate(pocket_atoms):
        for ax2, a2 in enumerate(binder_atoms):
            dists[ax1, ax2] = min(flat([[f.distance(at1, at2) for at1 in a1] for at2 in a2]))
    return dists

def get_features_res1(fragm, j, pocket_res, keywords, ss):
    fragm_features = None
    binder_pose, pocket_pose = None, None
    pocket_res_fragm = get_pocket_res_fragm(fragm, j, pocket_res, keywords)
    if pocket_res_fragm is not None:
        pocket_res_num_dict = {int(i.amn_number): idx for idx, i in enumerate(pocket_res_fragm)}
        file_name1 = prefix + '_' + j[0].chain_obj.label + str(j[0].amn_number)
        j_amns_pdb = ''.join([amn.amn_pdb for amn in j])
        pocket_res_pdb = ''.join([amn.amn_pdb for amn in pocket_res_fragm])
        cpx_pdb = j_amns_pdb + pocket_res_pdb
        f.writefile(prefix + '/structures/' + file_name1 + '_cpx.pdb', cpx_pdb)
        cpx_pose = pose_from_pdb(prefix + '/structures/' + file_name1 + '_cpx.pdb')
        bin_len = max([ax for ax in range(cpx_pose.total_residue()+1) if cpx_pose.pdb_info().chain(ax) == cpx_pose.pdb_info().chain(1)])
        binder_pose, pocket_pose = f.separate_frags(cpx_pose, bin_len)
        binder_pose = make_all_gly(binder_pose, keywords)

        pocket_atoms, binder_atoms = [f.get_coords(x) for x in [pocket_pose, binder_pose]]
        pocket_amns = [ad.amn_mapping[rev_ad[amn]] for amn in pocket_pose.sequence()]
        dists = calc_dist(pocket_atoms, binder_atoms)
        pocket_atoms = [[pocket_amns[ax], ax + 7, a] for ax, a in enumerate(pocket_atoms)]
        fragm_features = {'secondary': ss[fragm[0] - 1: fragm[0] + 6], 'homo': keywords['interf_type'],
                'secondary_p': [ss[int(a.amn_number) - 1] for a in pocket_res_fragm],
                'set2': [pocket_res_num_dict[int(a.amn_number)] + 7 for a in pocket_res_fragm],
                'pocket_struct': pocket_atoms, 'binder_struct': binder_atoms, 'dists': dists.tolist()}
    return pocket_res_fragm, fragm_features, binder_pose, pocket_pose

def get_features(keywords):
    pdb = keywords['pdb']
    prefix = keywords['prefix']
    make_folders(prefix)
    protein, all_amn, ss = get_protein(pdb)
    res1_int, res_int = get_binder_res_num(keywords)
    if (res1_int is None) and (res_int is None):
        print('Fragments are not specified')
        exit()

    binder_res = [a for a in  all_amn if int(a.amn_number) in np.unique(flat(res1_int + res_int))]
    protein.find_atoms_interactions('seq', 20, binder_res)
    pocket_res = get_pocket_res(binder_res, all_amn, keywords)

    features = []
    for fx, fragm in enumerate(res1_int):
        j = [a for a in binder_res if int(a.amn_number) in fragm]
        pocket_res_fragm, features_fragm, binder_pose, pocket_pose = get_features_res1(fragm, j, pocket_res, keywords, ss)
        if features_fragm is None:
            continue
        file_name1 = prefix + '_' + j[0].chain_obj.label + str(fragm[0])
        features_fragm['pdb_binder'] = file_name1
        features_fragm['p_f'] = -1
        features.append(features_fragm)
        binder_pose.dump_pdb(prefix + '/structures/' + file_name1 + 'binder.pdb')
        pocket_pose.dump_pdb(prefix + '/structures/' + file_name1 + 'pocket.pdb')

    if res_int is not None:
        all_binder_res = []
        all_pock_res = []
        for intx, interval in enumerate(res_int):
            #print(interval)
            int_res = [a for a in binder_res if int(a.amn_number) in interval]
            file_name1 = prefix + '_' + int_res[0].chain_obj.label + str(interval[0])
            for fx in range(len(interval[:-5])):
                fragm = interval[fx: fx+6]
                j = [a for a in binder_res if int(a.amn_number) in fragm]
                pocket_res_fragm, features_fragm, binder_pose, pocket_pose = get_features_res1(fragm, j, pocket_res, keywords, ss)
                if features_fragm is None:
                    continue
                features_fragm['pdb_binder'] = file_name1
                features_fragm['p_f'] = fx
                features.append(features_fragm)
                all_binder_res += [a for a in j if a not in all_binder_res]
                all_pock_res += [a for a in pocket_res_fragm if a not in all_pock_res]
            all_binder_res.sort(key=lambda x: int(x.amn_number))
            all_pock_res.sort(key=lambda x: int(x.amn_number))
            binder_pdb = ''.join([amn.amn_pdb for amn in all_binder_res])
            pock_pdb = ''.join([amn.amn_pdb for amn in all_pock_res])
            f.writefile(prefix + '/structures/' + file_name1 + 'binder.pdb', binder_pdb)
            f.writefile(prefix + '/structures/' + file_name1 + 'pocket.pdb', pock_pdb)
            f.writefile(prefix + '/structures/' + file_name1 + '_cpx.pdb', binder_pdb + pock_pdb)
    return features

input_file = sys.argv[1]
keywords = f.keywords(input_file)
prefix = keywords['prefix']
features = get_features(keywords)
if features != []:
    data= mk.get_data(features)
    f.writejson(prefix + '/' + prefix  + '_data_aas.json', data)
else:
    print('Operation has completed with error')
