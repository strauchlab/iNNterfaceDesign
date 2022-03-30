# iNNterfaceDesign
This package provides a framework for one-sided design of protein-protein interfaces based on features of protein receptor. This novel method is based on pipeline of neural networks processing features of the protein surface and producing the relevant binders and motifs for the target. The method is extremely quick and does not imply scanning of libraries with native PPI. 

The figure below demonstrate performance of the method on SARS-CoV-2 receptor binding domain in case of generating helical binders. Crystal structures of all generated binders for the target can be found in folder "SARS-CoV-2_design_examples" for more clear overview of capabilities of the method. 



https://user-images.githubusercontent.com/29002564/144135615-6736fb56-fcc0-488b-a8ef-76492e07a841.mp4






https://user-images.githubusercontent.com/29002564/148499738-37256419-56ed-4e27-ab71-90b9ae318f60.mp4





The framework consists of following main neural network models:
1) PepBB generating 6-residue backbones of binders;
2) BepBBE elongating initial backbones;
3) PepSeP1/PepSep6 designing amino acid sequences for the backbones.

All these methods can be used separately.
The software is developed for Linux-based systems.


The following software and python packages have to be installed  in order to run iNNterfaceDesign.
1. python v3.7;
2. PyRosetta-4 2019;
3. Tensorflow v2.1.0;
4. h5py v.2.10.0.
5. NumPy v.1.19.1

A manual to the program can be found here, in "Manual.pdf" file. The manual contains instractions for both iNNterfaceDesign as a whole and PepSep methods only separately. Besides, video showing installation process and test runs are uploaded to "videos" folder or can be watched using these links:

https://www.linkedin.com/posts/raulia-syrlybaeva-a9b4737b_innterfacedesign-activity-6914257560657113088-deeJ?utm_source=linkedin_share&utm_medium=member_desktop_web

https://www.linkedin.com/posts/raulia-syrlybaeva-a9b4737b_innterfacedesign-activity-6914263775223111680-dt3a?utm_source=linkedin_share&utm_medium=member_desktop_web



### How do I reference this work?

Syrlybaeva R., Strauch E-M. Deep learning of Protein Sequence Design of Protein-protein Interactions.  
bioRxiv (2022) doi: 10.1101/2022.01.28.478262
