% MATLAB code "Gamma.m" computes the surface excess energy density (Gamma) as a function of surface elastic invariants and in-plane deformation for 7 face-centered cubic (FCC) metals (Ag, Al, Au, Cu, Ni, Pd and Pt).       
 
% The surface invariants represent the surface elastic properties, including the residual surface stress tensor (Sigma) and the surface elastic stiffness tensor (C), within a polar method.                    
                                                                         
% In this calculation, all the surface elastic invariants are predicted by the developed artificial neural network (ANN) models. The trained ANN models are stored in the file "Networks_FCC_FreeSurface.mat". Only the surface orientation (represented by two angles theta and phi) is considered as the input for the developed ANN models. For each studied material, there are 304 different surface configurations are considered in the database.
% Users can train their own ANN models with different ANN structures, using the provided code "TrainNetworks.m" based on the results from the semi-analytical calculation.    
% Function "Create_Database.m" is used to create the database of surface elastic invariants from the results based on semi-analytical calculation presented in Cartesian frame, which are stored in "Data_SemiAnalytic.mat". The database of surface elastic invariants with the form used for the training code is stored in "Database_FCC_FreeSurface.mat"                                                
                                                                         
% In the code, the output variables "Strain_Magnitude", "Gamma_Total", "Gamma_Stress" and "Gamma_Stiffness" indicate the magnitude of the applied in-plane strain, the total of Gamma, the contribution of Sigma for Gamma and the contribution of C for Gamma, respectively.             
% Meanwhile, a Figure of the evaluation of Gamma as a function of the applied in-plane strain magnitude is created, with the name of the form "Gamma_Strain Mode_on_Material_Surface orientation_with reference_Reference Principal Direction=Deflection Angle_degree and nu=Concentration Factor_Contribution or Not".                                                     
                                                                         
% For more details, please check in the reference paper.                  
                                                                         
% References                                                                                                                                   
% X. Chen, R. Dingreville, T. Richeton and S. Berbenni. Invariant surface elastic properties in FCC metals and their correlation to bulk properties revealed by machine learning methods. Journal of the Mechanics and Physics of Solids, accepted.  
