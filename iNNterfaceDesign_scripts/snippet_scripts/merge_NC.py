import os, numpy as np, sys, json, random
from modules import functions as f, geometry_functions as fm, Class_protein as cp, Amino_acid_dictionaries as ad

dir = "/home/raulia/targets/6vy4/"
dir_score = dir + '/6vy4_9binders/' 
dir_NC = dir + '/6vy4_HNCnew/binders/'
blist = os.listdir(dir_score)

blistu = np.unique(['_'.join(a.split('_')[:-5]) for a in blist if '_C_id.pdb' in a]).tolist()

for b in blistu[:]:
    
    blist_x = [a for a in blist if a.startswith(b+'_')]
    listN = [a for a in blist_x if '_N_' in a]
    listC = [a for a in blist_x if '_C_' in a]
    if (listN == []) or (listC==[]):
        continue
    for i1 in listN:
        i1_file = dir_score +  i1.split('_N_')[0] + "_N_id.pdb"
        pdb1 = cp.Protein(i1_file)
        amns1 = [a.amn_pdb for a in pdb1.chains[0].amns]
        name1 =  i1.split("/")[-1].split("_N_")[0] + "_"
        for i2 in listC:
            i2_file = dir_score +  i2.split('_C_')[0] + "_C_id.pdb"
            pdb2 = cp.Protein(i2_file)
            amns2 = [a.amn_pdb for a in pdb2.chains[0].amns[-3:]]
            pdb_amns = "".join(amns1 + amns2)
            name2 = "_".join(i2.split("/")[-1].split("_")[4:-1])  
            name = dir_NC + name1 + name2 + "_NC.pdb"
            f.writefile(name, pdb_amns)
        
