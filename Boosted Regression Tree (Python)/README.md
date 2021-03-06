# Python code "ML_FCC_FreeSurface.py" builds the Boosted Regression Tree (BRT) models within Sklearn package for predicting non-angular surface elastic invariants (Gamma0, T, R, T0, R0, T1, R1) of 7 face-centered cubic (FCC) metals (Ag, Al, Au, Cu, Ni, Pd and Pt). 

# Gamma0 is the intrinsic surface excess energy density. 
# T and R are invariants of residual surface stress tensor.
# T0, R0, T1 and R1 are invariants of surface stiffness tensor.

# The database for each surface elastic invariants is named by "FCC_FreeSurface_Name of Invariant.txt".

# In the database file, each column presents the considered material's features with the order described below. Then, each row represents a surface of one material. In total, there are 2128 different surface configurations.

# The considered material's features used for BRT models are the {100}<110> shear resistance G' (in GPa), the {001}<100> shear resistance G'' (in GPa), the bulk modulus K (in GPa), the anisotropy ratio (Zener coefficient) A, the lattice parameter a (in nm), the stacking fault energy SFE (in J/m2), the cohesive energy CHE (in eV), two angular parameters represents the surface orientation theta and phi (in degree).

# For more details, please check in the reference paper.

# References
# X. Chen, R. Dingreville, T. Richeton and S. Berbenni. Invariant surface elastic properties in FCC metals via machine learning methods. Submission to Journal of the Mechanics and Physics of Solids. 
