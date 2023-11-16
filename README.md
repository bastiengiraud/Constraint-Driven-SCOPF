# Constraint-Driven-SCOPF

Code related to the submission of

- B.N. Giraud, A. Rajaei, J.L. Cremer. "Constraint-Driven Deep Learning for N-k Security Constrained Optimal Power Flow". Submitted to PSCC 2024.

# Code structure

This repository contains the following parts:

- 'learn SCOPF create GLODF.ipynb' is the code used to generate the LODF matrices.
- 'learn SCOPF correction reduced code.ipynb' is the code used to test the 39-bus system.
- The requirements.txt file contains the required packages.
- The folder 'Data' contains the training and testing data used. 
- The folder 'DelftBlue' contains the codes used to test the 118-bus system.
- The folder 'LODFS' contains the LODFs used for the 39-bus system.
- The folder 'Security Assessment' contains the files for the second case study.
- The folder 'Support' contains the .py files which are imported in the main Jupyter Notebooks.
- The folder 'Test Systems' contains the excels with the case study data.
- The folder 'Trained Models' contains the weights of the trained models which can be used for inference.
