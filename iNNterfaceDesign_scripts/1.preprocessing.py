import os, numpy as np, sys
from modules import functions as f, geometry_functions as fm, make_input, Class_protein as cp
from modules import Amino_acid_dictionaries as ad
from pyrosetta import *
from pyrosetta.teaching import *
from pyrosetta.rosetta.core.pose import *
from pyrosetta.toolbox import *
from pyrosetta.bindings.pose import *
from pyrosetta.rosetta.core.scoring.sasa import *

init('-ignore_unrecognized_res T -no_his_his_pairE T -no_optH F -use_input_sc T -ex1 T -ex2 T -ignore_zero_occupancy F ')
flat = lambda x: [i for j in x for i in j]
tolist = np.ndarray.tolist
DSSP = pyrosetta.rosetta.protocols.moves.DsspMover()


def get_core_amns_from_pdb(amns):
    amns_dict = {a.amn_number: a for a in amns}
    atoms = flat([a.atoms_no_h for a in amns])
    atoms_xyz = [at.xyz for at in atoms]
    neighs = fm.neighs_dict(atoms_xyz, atoms_xyz)
    core = fm.find_core(atoms_xyz, atoms_xyz, neighs)
    atoms_core, count = np.unique([i.amn_obj.amn_number for idx, i in enumerate(atoms) if idx in core],
                                  return_counts=True)
    atoms_core_list = [[amns_dict[atoms_core[i]], len(amns_dict[atoms_core[i]].atoms_no_h) - count[i]] for i in
                       range(len(count))]
    atoms_core_amns = [i[0] for i in atoms_core_list if i[1] == 0]
    return atoms_core_amns

def make_folders(prefix):
    if os.path.isdir(prefix) == False:
        os.mkdir(prefix)
    if os.path.isdir(prefix + '/pockets') == False:
        os.mkdir(prefix + '/pockets')
    if os.path.isdir(prefix + '/binders') == False:
        os.mkdir(prefix + '/binders')

def get_protein(pdb, keywords):
    protein = cp.Protein(pdb + '.pdb')
    all_amns = flat([chain.amns for chain in protein.chains])
    clean_pose = pose_from_pdb(pdb + '.pdb')
    DSSP.apply(clean_pose)
    ss = clean_pose.secstruct()
    if len(ss) != len(all_amns):
        print(pdb, ': number of residues in Rodetta pose and initial structure do not match')
        sys.exit()
    if 'p_chains' in keywords:
        p_chains = [a.strip() for a in keywords['p_chains'].split(',')]
        all_amns = [a for a in all_amns if  a.chain_obj.label in p_chains]
    all_amns.sort(key=lambda x: int(x.amn_number))
    return protein, all_amns, ss, clean_pose

def get_core_amns(all_amns, keywords, clean_pose):
    prefix = keywords['prefix']
    if 'add_residues' in keywords:
        core_amns = f.readjson(prefix + '/' + prefix + '_2b.json')[0]['core']
        core_amns = [i for i in all_amns if int(i.amn_number) in core_amns]
    elif ('get_surf_res' in keywords) and (keywords['get_surf_res'] == 'innf_core'):
        core_amns = get_core_amns_from_pdb(all_amns)
    else:
        sasac = np.array(rel_per_res_sc_sasa(clean_pose))
        core = [ax+1 for ax, a in enumerate(sasac) if a <= 0.1]
        core_amns  = [i for i in all_amns if int(i.amn_number) in core]
    return core_amns

def get_prev_binders(keywords):
    b_names = None
    prefix = keywords['prefix']
    if ('binders_list' in keywords) and (keywords['binders_list'] != ''):
        b_names = f.readfile(keywords['binders_list'], 't')
        b_names = [a.strip() for a in b_names.split(',')]
    elif os.path.exists(prefix + '/' + prefix + '_names.json'):
        b_names = f.readjson(prefix + '/' + prefix + '_names.json')
    return b_names

def get_anchor_res(all_amns, all_ca_atoms, surf_amns, keywords):
    binder_xyz = None
    if ('add_residues' in keywords) and (keywords['add_residues'] == 'True'):
        b_names = get_prev_binders(keywords)
        if b_names is None:
            print('The list of binder to prolongate is not provided')
            exit()
        ctrl_resi = []
        binder_xyz = []
        for b_name in b_names:
            if os.path.exists(prefix + '/binders_id/' + b_name[:-4] + '_id.pdb'):
                binder = cp.Protein(prefix + '/binders_id/' + b_name[:-4] + '_id.pdb')
                b_xyz = np.array([[x.xyz for x in a.atoms_no_h[:4]] for a in binder.chains[0].amns])
                b_xyz1 = b_xyz[0][1]
                b_xyz2 = b_xyz[-1][1]
                close_amns1 = sorted([[a.amn_obj, f.distance(b_xyz1, a.xyz)] for a in all_ca_atoms], key=lambda x: x[1])[0][0]
                close_amns2 = sorted([[a.amn_obj, f.distance(b_xyz2, a.xyz)] for a in all_ca_atoms], key=lambda x: x[1])[0][0]
                ctrl_resi += [close_amns1.amn_number, close_amns2.amn_number]
                binder_xyz.append([b_name, b_xyz, close_amns1.amn_number, close_amns2.amn_number])
        ctrl_resi = [a for a in all_amns if a.amn_number in ctrl_resi]

    elif 'anchor_res' in keywords:
        anums = flat(f.get_explicit_numbers(keywords['anchor_res']))
        ctrl_resi = [i for i in all_amns if int(i.amn_number) in anums]
    elif 'chain' in keywords:
        chains = [a.strip() for a in keywords['chain'].split(',')]
        ctrl_resi = [i for i in all_amns if (i.chain_obj.label in chains)] # and (i in surf_amns)]
    else:
        ctrl_resi = surf_amns
    return ctrl_resi, binder_xyz

def get_ld_ldd(keywords, all_amns):
    def get_ld_xyz(inp, amns, type):
        ld = None
        if type in inp:
            ld_amn = flat(f.get_explicit_numbers(inp[type]))
            ld_t = [a for a in amns if int(a.amn_number) in ld_amn]
            if ld_t != []:
                ld = [a.atoms_no_h[1].xyz for a in ld_t]
        return ld

    ld1 = get_ld_xyz(keywords, all_amns, 'pos_res1')
    ld2 = get_ld_xyz(keywords, all_amns, 'pos_res2')
    ld = [np.array(a) if a is not None else None for a in [ld1, ld2]]

    ld1d = get_ld_xyz(keywords, all_amns, 'pos_res1d')
    ld2d = get_ld_xyz(keywords, all_amns, 'pos_res2d')
    ldd = [np.array(a) if a is not None else None for a in [ld1d, ld2d]]
    return ld, ldd

def get_pocket_amns(all_amns, all_ca_atoms, core_amns, j):
    thr = 6
    neigh_pp = fm.icosahedron(np.array([0, 0, 0]), 2, 3.42) + np.array(j.atoms_no_h[1].xyz)
    neigh_pp1 = [sorted([[n, f.distance(n, m.xyz)] for m in all_ca_atoms], key=lambda x:x[1])[0] for n in neigh_pp]
    neigh_pp = [n[0] for n in neigh_pp1 if n[1] >= thr]
    if len(neigh_pp) == 0:
        neigh_pp = [n[0] for n in neigh_pp1 if n[1] >= thr - 1]
        if len(neigh_pp) == 0:
            return

    neigh_p = [sorted([[m.amn_obj, f.distance(n, m.xyz)] for n in neigh_pp], key=lambda x:x[1])[0] for m in all_ca_atoms]
    neigh_pa = [n for n in neigh_p if (n[1] <= 11.5) and (n[0] not in core_amns) and (
            f.distance(j.atoms_no_h[1].xyz, n[0].atoms_no_h[1].xyz) <= 20)]
    neigh_pa.sort(key=lambda x: x[-1])
    neigh_pa = neigh_pa[:30]
    neigh_pa = [n[0] for n in neigh_pa]
    neigh_ps = [a for a in all_amns if (a not in neigh_pa) and (int(a.amn_number) in flat(
        [[int(i.amn_number) - 1, int(i.amn_number) + 1] for i in neigh_pa]))]
    neigh_ps = [
        sorted([[a, f.distance(a.atoms_no_h[1].xyz, n.atoms_no_h[1].xyz)] for n in neigh_pa], key=lambda x:x[1])[0] for
        a in neigh_ps]
    neigh_pa += [n[0] for n in neigh_ps if n[1] < 4.5]
    neigh_pa = [sorted([[m.amn_obj, f.distance(n.atoms_no_h[1].xyz, m.xyz)] for n in neigh_pa], key=lambda x:x[1])[0]
                for m in all_ca_atoms]
    neigh_pa = [n for n in neigh_pa if (n[1] <= 8)]
    neigh_pa.sort(key=lambda x: x[-1])
    neigh_pa = neigh_pa[:48]
    neigh_pa = [n[0] for n in neigh_pa]
    return neigh_pa

def get_features(keywords):
    pdb = keywords['pdb']
    prefix = keywords['prefix']
    interf_type = keywords['interf_type']
    interf_res = None if ('interf_res' not in keywords) else f.get_explicit_numbers(keywords['interf_res'])

    make_folders(prefix)
    protein, all_amns, ss, clean_pose = get_protein(pdb, keywords)
    core_amns = get_core_amns(all_amns, keywords, clean_pose)
    surf_amns = [i for i in all_amns if i not in core_amns]
    all_atoms = flat([amn.atoms_no_h for amn in all_amns])
    all_ca_atoms = [a for a in all_atoms if a.type == 'CA']
    ctrl_resi, binder_xyz = get_anchor_res(all_amns, all_ca_atoms, surf_amns, keywords)
    ld, ldd = get_ld_ldd(keywords, all_amns)
    data = []
    for jx, j in enumerate(ctrl_resi[:]):
        ch_label = j.chain_obj.label
        pocket_amns = get_pocket_amns(all_amns, all_ca_atoms, core_amns, j)
        if (pocket_amns is None) or (j not in pocket_amns):
            continue

        set1 = [pocket_amns[a].amn_number for a in range(len(pocket_amns))]
        if len(pocket_amns) >= 24:
            pocket_amns.sort(key=lambda x: int(x.amn_number))
            pocket_amns_pdb = ''.join([amn.amn_pdb for amn in pocket_amns])
            file_name = prefix + '_' + ch_label + str(j.amn_number)
            f.writefile(prefix + '/pockets/' + file_name + 'pocket.pdb', pocket_amns_pdb)
            pocket_atoms = [amn.atoms_no_h for amn in pocket_amns]
            pocket_atoms = [[ad.amn_mapping[a[0].amn_obj.type], a[0].amn_obj.amn_number, [at.xyz for at in a]] for ax, a
                            in enumerate(pocket_atoms)]   
            data_dir= {'core': [int(a.amn_number) for a in core_amns], 'pdb': file_name, 'interf_type': interf_type, 'ld': ld, 'ldd': ldd,
                         'central_resi': int(j.amn_number), 'pocket_struct': pocket_atoms, 'secondary_p_all': ss,
                        'secondary_p': [ss[int(a.amn_number) - 1] for a in pocket_amns], 'set1': set1, 'ppi': interf_res}
            if 'add_residues' in keywords:
                b_data = [a for a in binder_xyz if j.amn_number in a[2:]]
                for b_d in b_data:
                    data_dir2 = dict(data_dir)
                    kx = 0 if j.amn_number == b_d[2] else 1
                    data_dir2['binder_end'] = kx
                    if kx == 0:
                        data_dir2['binder_name'] = b_d[0][:-4]
                        data_dir2['binder_xyz'] = b_d[1][:4].tolist()
                    else:
                        data_dir2['binder_name'] = b_d[0][:-4]
                        data_dir2['binder_xyz'] = b_d[1][-4:].tolist()
                    data.append(data_dir2) 
            else: 
                data.append(data_dir) 
    return data


input_file = sys.argv[1]
keywords = f.keywords(input_file)
prefix = keywords['prefix']
features = get_features(keywords)
if features != []:
    data1, data2 = make_input.make_data(features)
    f.writejson(prefix + '/' + prefix + '_b.json', data1)
    f.writejson(prefix + '/' + prefix + '_2b.json', data2)
else:
    print('Operation has completed with error')


