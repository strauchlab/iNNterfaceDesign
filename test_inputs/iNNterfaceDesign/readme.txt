1. Example of input for design of most probable (top) 6-residue motifs for a few anchor residues

pdb: 3ztj_ABCDEF.pdb
input file: input_top

Expected output:
folder: "3ztj_top"
designed amino acid sequences in 3ztj_top/3ztj_top_aas.json file
deisgned backbones in 3ztj_top/binders_id folder
relaxed motifs complexed with binding sites in 3ztj_top/binders_rel folder

2. Example of inputs for design of 12-residue motifs which 6-residue central parts are generated for 10 top ranked positions and 5 most probable secondary structure sequences

pdb: 3ztj_AB.pdb
input file1: input_10_5
input file2: input_10_5_a

Expected output:
folder: "3ztj"
designed amino acid sequences in 3ztj/3ztj_aas.json file
deisgned backbones: 3ztj_top/binders_id/*NC*
relaxed motifs complexed with binding sites in 3ztj/binders_rel folder


