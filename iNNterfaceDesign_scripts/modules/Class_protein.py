import numpy as np
from  modules import Amino_acid_dictionaries as ad, functions as f


tolist = np.ndarray.tolist
flat = lambda x: [i for j in x for i in j]
unq = lambda x: np.array(np.unique(x))


def toint(j):
    j = j.strip()
    if j[0] != '-':
        j = int(j) if j.isdigit() else int(j[:-1])
    else:
        j = j[1:]
        j = int(j) if j.isdigit() else int(j[:-1])
        j = -j
    return j


class Protein(object):

    def __init__(self, path):
        self.pdb = f.readfile(path, 'l')
        self.pdb = [i for i in self.pdb if (i[17:20] != 'HOH') and (i.startswith("CONECT") == False)]
        pyr = [idx for idx, i in enumerate(self.pdb) if 'All scores below are weighted scores, not raw scores' in i]
        if pyr != []:
            self.pdb = self.pdb[:pyr[0]]
        self.atom_lines = [at for at in self.pdb if at.startswith('ATOM') or at.startswith('HETATM')]
        self.get_chains()

    def get_data_annotation(self):
        """
        resol = [i for i in self.pdb if 'REMARK   2 RESOLUTION.' in i][0]
        try:
            if 'AT' in resol:
                self.resol = float(resol[:resol.rfind('A RESOLUTION.')].split()[-1])
            else:
                self.resol = float(resol[resol.find('RESOLUTION.') + 11: resol.find('ANG')].strip())
        except:
            self.resol = 'unknown'
        """
        title = ' '.join(''.join([i[10:-1] for i in self.pdb if i.startswith('TITLE')]).split()).capitalize()
        self.title = title
        compd = ''.join([i[:-1] for i in self.pdb if i.startswith('COMPND')]).split('MOL_ID: ')
        compd = [i.split(';') for i in compd]
        compd = [[j.strip() for j in i if ('CHAIN:' in j) or ('MOLECULE:' in j)] for i in compd]
        compd = [[i[0][i[0].find('MOLECULE:')+10:], i[1][i[1].find('CHAIN')+7:].split(',')] for i in compd if 'CHAIN' in ''.join(i)]
        chains = flat([i[1] for i in compd])
        chains = {i.strip():[k[0] for k in compd if i in k[1]][0]  for i in chains}
        self.molecules = chains

    def get_chains(self):
        chains = np.unique([i[20:22] for i in self.atom_lines if i[20:22] != ''])
        self.chains = [Chain(self, [at for at in self.atom_lines if at[20:22] == i], i) for i in chains]

    def find_chains_interactions(self, it, thr):
        atoms_ca = [[a for a in i.atoms_no_h if (a.type == 'CA') or (a.amn_obj.categ == 'ligand')]  for i in self.chains]
        if it == 'interface_all':
            atoms = [i.atoms_no_h for i in self.chains]

        for chx1, ch1 in enumerate(atoms_ca):
            for chx2, ch2 in enumerate(atoms_ca):
                if chx1 < chx2:
                    for idx, i in enumerate(ch1):
                        for jdx, j in enumerate(ch2):
                            d = f.distance(i.xyz, j.xyz)
                            if d <= 16:
                                name_i = i.amn_obj.amn_name
                                name_j = j.amn_obj.amn_name
                                if it == 'interface_ca':
                                    if (name_j not in i.amn_obj.close) or (i.amn_obj.close[name_j][-1] > d):
                                        i.amn_obj.close[name_j] = [j.amn_obj.chain_obj.label, round(d, 3)]
                                        j.amn_obj.close[name_i] = [i.amn_obj.chain_obj.label, round(d, 3)]
                                elif it == 'interface_all':
                                    atoms_amn_i = [a for a in atoms[chx1] if i.amn_obj.amn_name == a.amn_obj.amn_name]
                                    atoms_amn_j = [a for a in atoms[chx2] if j.amn_obj.amn_name == a.amn_obj.amn_name]
                                    for idx1, i1 in enumerate(atoms_amn_i):
                                        for jdx2, j2 in enumerate(atoms_amn_j):
                                            d = f.distance(i1.xyz, j2.xyz)
                                            if d <= thr:
                                                if (name_j not in i.amn_obj.close) or (i.amn_obj.close[name_j][-1] > d):
                                                    i.amn_obj.close[name_j] = [j.amn_obj.chain_obj.label, i1.type,
                                                                               j2.type, round(d, 3)]
                                                    j.amn_obj.close[name_i] = [i.amn_obj.chain_obj.label, j2.type,
                                                                               i1.type, round(d, 3)]


    def find_atoms_interactions(self, it, thr, seq=[]):
        if it == 'all_int':
            atoms1 = flat([i.atoms_no_h for i in self.chains])
            atoms2 = atoms1
        elif it == 'seq':
            atoms1 = flat([i.atoms_no_h for i in self.chains])
            atoms2 = flat([i.atoms_no_h for i in seq])
        elif it == 'seq_ca':
            atoms1 = flat([i.atoms_no_h for i in self.chains])
            atoms1 = [a for a in atoms1 if a.type == 'CA']
            atoms2 = flat([i.atoms_no_h for i in seq])
        elif it == 'seq_bb':
            atoms1 = flat([i.atoms_no_h for i in self.chains])
            atoms1 = [a for a in atoms1 if a.type in ['N', 'CA', 'C', 'O']]
            atoms2 = flat([i.atoms_no_h for i in seq])
            atoms2 = [a for a in atoms2 if a.type in ['N', 'CA', 'C', 'O']]


        atoms1_ca = [a for a in atoms1 if a.type == 'CA'] + [a for a in atoms1 if a.amn_obj.categ == 'ligand']
        atoms2_ca = [a for a in atoms2 if a.type == 'CA']
        for idx, i in enumerate(atoms1_ca):
            for jdx, j in enumerate(atoms2_ca):
                d = f.distance(i.xyz, j.xyz)
                if d <= 25:
                    name_i = i.amn_obj.amn_name
                    name_j = j.amn_obj.amn_name
                    if name_i == name_j: #((name_i.split('_')[0] == name_j.split('_')[0]) and (abs(int(name_i.split('_')[1]) - int(name_j.split('_')[1])) == 1)) or (
                        continue
                   
                    #print(name_i, name_j, d)
                    atoms_amn_i = [a for a in atoms1 if i.amn_obj.amn_name == a.amn_obj.amn_name]
                    atoms_amn_j = [a for a in atoms2 if j.amn_obj.amn_name == a.amn_obj.amn_name]
                    for idx1, i1 in enumerate(atoms_amn_i):
                        for jdx2, j2 in enumerate(atoms_amn_j):
                            d = f.distance(i1.xyz, j2.xyz)
                            if d <= thr:
                                i1.close += 1
                                j2.close += 1
                                if (name_j not in i.amn_obj.close) or (i.amn_obj.close[name_j][-1] > d):
                                    i.amn_obj.close[name_j] = [j.amn_obj.chain_obj.label, i1.type, j2.type, round(d, 3)]
                                    j.amn_obj.close[name_i] = [i.amn_obj.chain_obj.label, j2.type, i1.type, round(d, 3)]

    def standardorientation(self, targ_amns = 0):

        atoms = flat([ch.atoms for ch in self.chains])
        atoms_xyz = np.array([i.xyz for i in atoms])
        atoms_xyz, inertiavectors = f.standardorientation(atoms_xyz)

        for atx, a in enumerate(atoms):
            a.xyz = atoms_xyz[atx]
            xyz_str = ''.join(["{:8.3f}".format(x) for x in a.xyz])
            a.string = a.string[:30] + xyz_str + a.string[54:]
        return inertiavectors

    def translate_to_center(self, tr = None, targ_amns = 0):
        atoms = flat([ch.atoms for ch in self.chains])
        atoms_xyz = np.array([i.xyz for i in atoms])
        if tr is None:
            tr = atoms_xyz.mean(axis=0)
        atoms_xyz -= tr
        for atx, a in enumerate(atoms):
            a.xyz = atoms_xyz[atx]
            xyz_str = ''.join(["{:8.3f}".format(x) for x in a.xyz])
            a.string = a.string[:30] + xyz_str + a.string[54:]
        return tr

    def align_point(self, p1, p2=None, a1 = None, ortha = None, c=None):
        if c is None:
            c= np.array([0, 0, 0])
        atoms = flat([ch.atoms for ch in self.chains])
        atoms_xyz = np.array([i.xyz for i in atoms])
        if a1 is None:
            if ortha is None:
                norm = p1 / np.linalg.norm(p1)
                ortha = np.cross(p2, norm)
            a1 = f.dihedral(np.array([p1, c, ortha, p2]))
        if a1 == a1:
            atoms_xyz = f.rotate_dihedral(a1, ortha, c, atoms_xyz)
        for atx, a in enumerate(atoms):
            a.xyz = atoms_xyz[atx]
            xyz_str = ''.join(["{:8.3f}".format(x) for x in a.xyz])
            a.string = a.string[:30] + xyz_str + a.string[54:]
        return a1, ortha

class Chain(object):

    def __init__(self, protein_obj, pdb, label):
        self.protein_obj, self.pdb, self.label = protein_obj, pdb, label.strip()
        self.make_aminoacids()
        self.amns = [Aminoacid(self, i) for i in self.amn_indexes]
        self.atoms = [a for b in [i.atoms for i in self.amns] for a in b]
        self.atoms_no_h = [a for b in [i.atoms_no_h for i in self.amns] for a in b]
        self.atoms_xyz_no_h = np.array([a.xyz for a in self.atoms_no_h])
        self.center = np.mean(self.atoms_xyz_no_h, axis=0)

    def make_aminoacids(self):
        amn_indexes = [l[22:27].strip() for l in self.pdb]
        amn_indexes = tolist(np.unique(np.array([amn_indexes])))
        amn_indexes_neg = [i for i in amn_indexes if i[0] == '-']
        amn_indexes.sort(key=lambda x: toint(x))
        self.amn_indexes = amn_indexes
        if amn_indexes_neg != []:
            self.amn_indexes_neg = True
        else:
            self.amn_indexes_neg = False
 
    def get_fragments(self):
        def make_helix_sheet(idx, frg, text, label):
            if frg == 'HELIX':
                fragments = [l for l in text if l.startswith(frg + ' ') and (l[19:20] == label)]
                fragments = [[toint(i[21:25]), toint(i[33:37])] for i in fragments]
            else:
                fragments = [l for l in text if l.startswith(frg + ' ') and (l[21:22] == label)]
                fragments = [[toint(i[22:26]), toint(i[33:37])] for i in fragments]
            if fragments !=[]:
                fragments = np.unique(fragments, axis=0).tolist()
                fragments = [[j for j in idx[1:] if toint(j) in range(k[0], k[1] + 1)] for k in fragments]
                fragments = [i for i in fragments if len(i) > 0]
            return fragments

        def make_loop(loop):
            loops = []
            while len(loop) > 0:
                seq = [loop[0]]
                loop = loop[1:]
                for adx, a in enumerate(loop):
                    if toint(a) - toint(seq[-1]) <= 1:
                        seq.append(a)
                        loop = loop[1:]
                    else:
                        loops.append(seq)
                        break
            loops.append(seq)
            return loops

        helix = make_helix_sheet(self.amn_indexes, 'HELIX', self.protein_obj.pdb, self.label)
        sheet = make_helix_sheet(self.amn_indexes, 'SHEET', self.protein_obj.pdb, self.label)
        occ = [x for y in helix + sheet for x in y]
        loop = [i for i in self.amn_indexes if i not in occ]
        if loop != []:
            loops = make_loop(loop)
            self.loop = loops
        else:
            self.loop = []
        self.helix = helix
        self.sheet = sheet

    def make_secondary_seq(self):
        merged = self.loop + self.helix + self.sheet
        merged.sort(key=lambda x: toint(x[0]))
        secondary = [[[i.secondary for i in self.amns if i.amn_number == j][0] for j in s][0] for s in merged]
        self.merged = merged
        self.secondary = secondary

    def find_chain_surface(self):
        from itertools import product
        from sklearn.cluster import DBSCAN

        def find_neighs(x, y, z, n, limits):
            xa, ya, za = list(range(x - n, x + n)), list(range(y - n, y + n)), list(range(z - n, z + n))
            cube_neighs = [list(i) for i in product(xa, ya, za)]
            neighs = [i for i in cube_neighs if f.distance(i, [x, y, z]) <= n]
            return neighs

        atoms_xyz = self.atoms_xyz_no_h
        center = self.center
        atoms_xyz = (atoms_xyz - center)/1.5
        min_xyz = atoms_xyz.min(axis=0)
        atoms_xyz -= min_xyz
        limits = np.ceil(np.max(atoms_xyz, axis=0)).astype(int)

        all_cells = [list(i) for i in product(list(range(limits[0])), list(range(limits[1])), list(range(limits[2])))]
        occupied = []
        for i in atoms_xyz:
            i = np.floor(i).astype(int)
            if i.tolist() not in occupied:
                occupied.append(i.tolist())
        occupied_vwd = [find_neighs(cell[0], cell[1], cell[2], 2, limits) for cell in occupied]
        occupied_vwd = np.unique([cell for atom in occupied_vwd for cell in atom], axis=0).tolist()
        empty_vwd = [i for i in all_cells if i not in occupied_vwd]

        db = DBSCAN(eps=2.0, min_samples=30, metric='euclidean')
        indexes = np.where(db.fit_predict(np.array(empty_vwd)) == 0)
        occupied_vwd += [i for idx, i in enumerate(empty_vwd) if idx not in indexes[0]]

        for amn in self.amns:
            amn_xyz = np.floor((np.array([a.xyz for a in amn.atoms_no_h]) - center)/1.5 - min_xyz).astype(int)
            amn_neigh = [find_neighs(cell[0], cell[1], cell[2], 3, limits) for cell in amn_xyz]
            amn_neigh = np.unique([cell for atom in amn_neigh for cell in atom], axis=0).tolist()
            amn_neigh_empty = len(amn_neigh) - len([a for a in amn_neigh if a in occupied_vwd])
            amn.empty_sp = amn_neigh_empty
        self.center = center

class Aminoacid(object):

    def __init__(self, chain_obj, number):
        self.chain_obj, self.amn_number = chain_obj, number
        self.amn_name = '_'.join([self.chain_obj.label, self.amn_number])
        self.close = {}
        self.get_atoms()
        self.empty_sp = 0
        
    def amn_secondary_structure(self):
        if self.amn_number in flat(self.chain_obj.helix):
            self.secondary = 'h'
        elif self.amn_number in flat(self.chain_obj.sheet):
            self.secondary = 's'
        elif self.amn_number in flat(self.chain_obj.loop):
            self.secondary = 'l'
        else:
            print('Error: secondary structure is not assigned')

    def get_atoms(self):
        atoms = [i for i in self.chain_obj.pdb if i[22:27].strip() == self.amn_number]
        amn_numbers = [i[22:27].strip() for i in atoms]
        if len(amn_numbers) > 1:
            atoms = [i for i in atoms if i[22:27].strip() == amn_numbers[0]]

        type = [a[16:20].strip() for a in atoms]
        if len(unq(type)) != 1:
            types4 = [i for i in unq(type) if len(i) != 3]
            atoms = [i for i in atoms if (i[16:17] == types4[0][0]) or (i[16:17] == ' ')]
            self.type = type[0][-3:]
        else:
            if len(type[0]) == 3:
                self.type = type[0]
            else:
                self.type = type[0][-3:]

        if atoms[0].startswith('ATOM') and (self.type in ad.amn_mapping1letter):
            self.categ = 'normal'
            self.type1l = ad.amn_mapping1letter[self.type]
            ns = [atx for atx, at in enumerate(atoms) if at[12:15] == ' N ']
            if len(ns) > 1:
                atoms = atoms[:ns[1]]
        else:
            self.categ = 'ligand'

        self.atoms = [Atom(self, i) for i in atoms]
        self.atoms_no_h = [i for i in self.atoms if [j for j in i.type if j.isdigit() == False][0] != 'H']
        self.amn_pdb = ''.join([i.string for i in self.atoms])
        self.amn_pdb_no_h = ''.join([i.string for i in self.atoms if [j for j in i.type if j.isdigit() == False][0] != 'H'])


class Atom(object):

    def __init__(self, amn_obj, i):
        self.amn_obj, string = amn_obj, i
        self.num = i.split()[1]
        self.type = i[12:16].strip()
        self.element = i[76:78]
        self.xyz = [float(a) for a in [i[30:38], i[38:46], i[46:54]]]
        xyz_str = ''.join(["{:8.3f}".format(x) for x in self.xyz])
        self.string = string[:30] + xyz_str + string[54:]
        self.close = 0

