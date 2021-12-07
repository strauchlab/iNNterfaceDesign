# iNNterfaceDesign
This package provides a framework for one-sided design of protein-protein interfaces based on features of protein receptor. This novel method is based on pipeline of neural networks processing features of the protein surface and producing the relevant binders and motifs for the target. The method is extremely quick and does not imply scanning of libraries with native PPI. 

The figure below demonstrate performance of the method on SARS-CoV-2 receptor binding domain. Crystal structures of all generated binders for the rarget can be found in folder "SARS-CoV-2_design_examples" for more clar overview of capabilities of the method. If someone is interested in these designs from practical point of view, they can contact with us for collaborative work.



https://user-images.githubusercontent.com/29002564/144135615-6736fb56-fcc0-488b-a8ef-76492e07a841.mp4


https://user-images.githubusercontent.com/29002564/144131115-ec23f2c6-7231-4bcd-af4f-5e64d50bfbf0.mp4



The framework consists of following main neural network models:
1) PepBB generating 6-residue backbones of binders;
2) BepBBE elongating initial backbones;
3) PepSeP1/PepSep6 designing amino acid sequences for the backbones.

All these methods can be used separately.

The outputs include designed backbone in a docked pose and a designed amino acid sequence for it. Mapping of the amino acid sequence to the backbone and subsequent relaxation have to be done in third-party software, like Rosetta. Normal job should include filteing as well, since some produced binders are of poor quality.


The software is developed for Linux-based systems.
The following software and python packages have to be installed  in order to run iNNterfaceDesign.
1. python v3.7 or higher;
2. PyRosetta;
3. Tensorflow.

A manual to the program can be found here, in "Manual.docx" file. The manual contains instractions for both iNNterfaceDesign as a whole and PepSep methods only separate;y.
