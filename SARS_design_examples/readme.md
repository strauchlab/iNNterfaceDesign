Presented binders are 12-residue peptides having α-helix and loop structures  designed for SARS-CoV-2 receptor binding domain. Crystal structure of the target protein, cleaned and renumbered by clean_pdb.py script of Rosetta software, is provided in 6w41_C.pdb file.

The following types of the binders differing by their secondary structure are presented:  
a) 12-residue α-helices stored in an archive file called “H”;  
b) 12-residue loops in an archive file called “L”;  
c) 9-residue α-helices having additional 3 loop residues at the right end, stored in an archive file called “HL”;  
d) 9-residue α-helices having additional 3 loop residues at the left end, stored in an archive file called “LH”;  
e) 6-residue α-helices having additional 3 loop residues at both left and right ends, stored in an archive file called “LHL”.

### Generating of the binders was carried out as follows:  
Central α-helices and loops were modeled for all possible positions having probability higher than 2% using PepBB model. The designed backbones were swapped with most similar natural 6-residue backbone fragments from fragments library of the method.

The backbones were extended by generating 3 additional residues using three trained PepBBE models* for each side of the binder.  Swapping of extended pieces with natural fragments was not conducted here. If helix residues were used for the extension then the extended fragment was generates for the most probable position; the extension with loop residues was carried out for  three most probable positions. Threshold for probability for selection of positions is 2%. 

*We repeated training of PepBBE model 10 times during development of the method and 3 trained models with top performances were kept for future usage. The best models were set as default in the framework however there is opportunity to extend backbones using all these models getting 3 times more poses: despite similarity of outputs of the models in most cases they are not exactly same and corresponding amino acid sequence designs for them can differ. 

Amino acid sequences were designed  using PepSP6 method; the sequence designs were filtered using build-in filter excluding designs with low diversity of types of amino acids in sequences.
Designed binders  were relaxed using Rosetta software, details of these calculations can be found in AA paper. Resulted binders were filtered using Geometry, FragmentLookupFilter and Sasa filters of Rosetta software; binders resulting in buried ASP and GLU residues  of the target were also excluded.

Amino acid sequences were designed once more fore relaxed backbones of the binders; besides amino acid sequences were additionally generated as for homo-oligomeric PPIs just to compare results.
Therefore, each of the archive files mentioned above contains three folders: “1” for the initial binders, “2” for binders with redesigned sequences, “3” for binders with animo-acid sequences designed as for homo-oligomeric PPIs.

The following residues of the target protein were used as anchor residues for designs: 10,23,32-49,51-54,56-62,64,71,73,74,76,77,80-83,95,96,98,101,104,105,131,132,171,172,176,183-190,195.  These are surface residues in proximity of the binding domain.
