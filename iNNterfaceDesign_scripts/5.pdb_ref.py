import os, numpy as np, json, pandas as pd, sys
from modules import functions as f, geometry_functions as fm, Class_protein as cp
from modules import Amino_acid_dictionaries as ad
from shutil import copyfile
from pyrosetta import *
from pyrosetta.teaching import *
from pyrosetta.rosetta.protocols.relax import *
from pyrosetta.rosetta.core.pack.task import *
from pyrosetta.rosetta.core.pack.task.operation import *
from pyrosetta.rosetta.protocols.task_operations import *
from pyrosetta.rosetta.protocols.minimization_packing import *
from pyrosetta.rosetta.core.select.residue_selector import *
from pyrosetta.rosetta.core.pose import *
from pyrosetta.rosetta.protocols.denovo_design.movers import *
from pyrosetta.toolbox import *
from pyrosetta.rosetta.protocols.idealize import *
from pyrosetta.rosetta.protocols.constraint_generator import *
from pyrosetta.rosetta.core.scoring.constraints import *
from pyrosetta.rosetta.protocols.simple_moves import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.pack.task.operation import RestrictToSpecifiedBaseResidueTypes
from rosetta.utility import vector1_string
from pyrosetta.rosetta.core.scoring.hbonds import *
from pyrosetta.bindings.pose import *
from pyrosetta.rosetta.core.scoring.sasa import *

init('-ignore_unrecognized_res T -no_his_his_pairE T -no_optH F -use_input_sc T -ex1 T -ex2 T -detect_disulf F')
flat = lambda x: [i for j in x for i in j]
tolist = np.ndarray.tolist
np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)
pd.options.display.float_format = '{:,.2f}'.format
unq = np.unique

def sc():
    scorefxn = ScoreFunctionFactory.create_score_function("ref2015_cart.wts")
    return scorefxn

def sc_constr():
    scorefxn = ScoreFunctionFactory.create_score_function("ref2015_cart.wts")
    scorefxn.set_weight(coordinate_constraint, 1.0)
    return scorefxn

def constrain(pose, l1):
    la = list(range(1,l1+1))
    add_cst = AddConstraints()
    index_selector = ResidueIndexSelector(','.join([str(i) for i in la]))
    coordinate_cst = CoordinateConstraintGenerator()
    coordinate_cst.set_ca_only(True)
    coordinate_cst.set_residue_selector(index_selector)
    coordinate_cst.set_bounded(True)
    coordinate_cst.set_bounded_width(2)
    coordinate_cst.set_sd(2) 
    coordinate_cst.set_reference_pose(pose)
    add_cst.add_generator(coordinate_cst)
    return add_cst

def sc_scan():
    ddG_scorefxn = ScoreFunctionFactory.create_score_function("ref2015_cart.wts")
    ddG_scorefxn.set_weight(fa_atr, 0.44)
    ddG_scorefxn.set_weight(fa_rep, 0.07)
    ddG_scorefxn.set_weight(fa_sol, 1.0)
    ddG_scorefxn.set_weight(hbond_bb_sc, 0.5)
    ddG_scorefxn.set_weight(hbond_sc, 1.0)
    return ddG_scorefxn

def movemap(k, loop_amns):
    mm = MoveMap()
    mm.set_chi(True)
    if k==1:
        mm.set_jump(True)
        for i in loop_amns: #[1:-1]:
            mm.set_bb(i, True)
    return mm

def relax_mover(scorefxn, k, l1):
    restrict1 = RestrictToRepacking()
    tf = TaskFactory()
    tf.push_back(InitializeFromCommandline())
    tf.push_back(restrict1)

    mm = movemap(k, list(range(1, l1+1)))
    relax = FastRelax()
    relax.cartesian(True)
    relax.max_iter(100)
    relax.set_scorefxn(scorefxn)
    relax.set_task_factory(tf)
    relax.set_movemap(mm)
    return relax

def min_mover(scorefxn, k, l1):
    mm = movemap(k, list(range(1, l1+1)))
    relax = MinMover()
    relax.score_function(scorefxn)
    relax.min_type("lbfgs_armijo")
    relax.tolerance(1e-6)
    relax.cartesian(True)
    relax.max_iter(100)
    relax.movemap(mm)
    return relax

def interface_eval(scorefxn):
    iam = InterfaceAnalyzerMover()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_pack_separated(True)
    iam.set_pack_rounds(20)
    return iam

def ala_scan(pose, scorefxn, interf_chains, l1):
    relax = min_mover(scorefxn, 0, l1)
    iam = interface_eval(scorefxn)
    iam.set_interface(interf_chains)
    iam.apply(pose)
    dG = iam.get_interface_dG()
    scores = {}
    for ix in range(1, l1+1):
        res_i = pose.residue(ix).name()
        if (res_i[:3] not in ['ALA', 'GLY', 'PRO']):
            scan_pose = pose.clone()
            mutate_residue(scan_pose, ix, 'A')
            relax.apply(scan_pose)
            iam.apply(scan_pose)
            dGa = iam.get_interface_dG()
            score_i = dG - dGa
            scores[ix] = [res_i, score_i]
        else:
            scores[ix] = [res_i, 0]
    return scores


def mutate_loop(data, dir, dir1, dir2, dir3, k1):
    scorefxn = sc()
    iam = interface_eval(scorefxn)
    score_data = []
   
    for cx, cpx in enumerate(data[:]):
        name, seq = cpx[0][:-4], cpx[1]
        p_name = name + 'pocket.pdb' 
        print(p_name)
        pocket = pose_from_pdb(dir1 + p_name)
        loop1 = pose_from_pdb(dir2 + name + '_id.pdb')
        loop_cst = loop1.clone()
        unq_seq = np.unique(seq)
        for m in range(len(unq_seq)):
            #try:
            loop = loop1.clone()
            l1 = len(unq_seq[m])
            pt_m = '_' + str(m) #unq_seq[m] 
            append_pose_to_pose(loop, pocket)

            for k in range(1,l1+1):
                mutate_residue(loop, k, unq_seq[m][k-1])
            
            relax1 = relax_mover(scorefxn, 0, l1)
            relax2 = relax_mover(sc_constr(), 1, l1)
            
            loop.dump_pdb(dir2 + name + pt_m +  '_cpx.pdb')
            loop = pose_from_pdb(dir2 + name + pt_m + '_cpx.pdb')
            
            pocket_chains = np.unique([loop.pdb_info().chain(x) for x in range(l1, len(loop.sequence())+1)])
            interf_chains = '_'.join(['A', ''.join(pocket_chains)])
            cpx_res = [] 
            for r in range(3):
                loop_rel = loop.clone()
                relax1.apply(loop_rel)
                iam.set_interface(interf_chains)
                iam.apply(loop_rel)
                cpx_res.append([iam.get_interface_dG(), loop_rel])
            
            cpx_res.sort(key=lambda x: x[0])
            loop = cpx_res[0][1]
            cpx_res = []
            for r in range(3):
                loop_rel = loop.clone()
                v_root = VirtualRootMover()
                v_root.set_removable(1)
                cst = constrain(loop_cst, l1)
                v_root.apply(loop_rel)
                cst.apply(loop_rel)
                relax2.apply(loop_rel)
                iam.apply(loop_rel)
                cpx_res.append([iam.get_interface_dG(), loop_rel])
            
            cpx_res.sort(key=lambda x: x[0])
            loop_rel = cpx_res[0][1]

            ddG = ala_scan(loop_rel, scorefxn, interf_chains, l1)
            loop_rel.dump_pdb(dir3 + name + pt_m +  '_cpx.pdb')
            score_data.append([name, m, unq_seq[m], cpx_res[0][0], ddG])
               
    print(score_data)
    f.writejson(dir + '/scores/' + k1.split('/')[-1] + '.json', score_data)

def merge_json(dir):
    data =[] 
    cpxs = os.listdir(dir)
    print(cpxs)
    for i in cpxs:
        data_i = f.readjson(dir + '/' + i)
        data += data_i
        
    f.writejson(dir.split('/')[0] + '_scores.json', data)


k1 = sys.argv[1]
cpx = f.readfile(k1, 'l')
dir = cpx[0].split('/')[0]

cpx = [a.strip().split('/')[-1] for a in cpx[:]]
data = f.readjson(dir + '/' + dir + '_aas.json')
data = [a for a in data if a[0][:-4] + '_id.pdb' in cpx]

dir1 = dir + '/pockets/'
dir2 = dir + '/binders_id/'
dir3 = dir + '/binders_rel/'

mutate_loop(data, dir, dir1, dir2, dir3, k1)

#merge_json('6vy4_Hr9/scores/')

