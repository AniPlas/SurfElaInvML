# This Python code builds the Boosted Regression Tree (BRT)
# models within Sklearn package for predicting non-angular
# surface elastic invariants (Gamma0, T, R, T0, R0, T1, R1)
# of 7 face-centered cubic (FCC) metals (Ag, Al, Au, Cu, Ni,
# Pd and Pt). 
#
# Gamma0 is the intrinsic surface excess energy density. 
# T and R are invariants of residual surface stress tensor.
# T0, R0, T1 and R1 are invariants of surface stiffness tensor.
#
# The database for each surface elastic invariants is named 
# by "FCC_FreeSurface_Name of Invariant.txt".
#
# In the database file, each column presents the considered
# material's features with the order described below. Then,
# each row represents a surface of one material. In total,
# there are 2128 different surface configurations.
#
# The considered material's features used for BRT models are
# the {100}<110> shear resistance G' (in GPa),
# the {001}<100> shear resistance G'' (in GPa),
# the bulk modulus K (in GPa),
# the anisotropy ratio (Zener coefficient) A,
# the lattice parameter a (in nm),
# the stacking fault energy SFE (in J/m2),
# the cohesive energy CHE (in eV),
# two angular parameters represents the surface orientation
# theta and phi (in degree).
#
# For more details, please check in the reference paper.
#
# References
# X. Chen, R. Dingreville, T. Richeton and S. Berbenni.
# Invariant surface elastic properties in FCC metals via
# machine learning methods.
# Submission to Journal of the Mechanics and Physics of Solids. 
#
# Importe necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import joblib

# Load data function
def loadDataSet(fileName):
    numFact = len(open(fileName).readline().split(' '))-1
    # print(numFact)
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split(' ')
        for i in range(numFact):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# Separate the dataset into five group for Five-fold cross-validation
# 80% of database is used for training and 20% used for testing 
Test_set_list = [0.0,0.2,0.4,0.6,0.8,1];
Name_Test_List = ["0","20","40","60","80","100"];
# List of considered non-angular surface elastic invariants 
file_list = ["Gamma0","T","R","T0","T1","R0","R1"]
Label_list = ["Gamma0 (J/m^2)","T (J/m^2)","R (J/m^2)","T0 (J/m^2)","T1 (J/m^2)",
              "R0 (J/m^2)","R1 (J/m^2)"]    

clf_DTR = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 10,
                                               min_samples_split = 5))

# Set the hyper-parameters by cross-validation
# 'loss':['linear','square','exponential']
# 'learning_rate':[0.05,0.1,0.25,0.5,0.75,1]
# 'n_estimators':[200,250,300,350,400,450]
tuned_parameters = {'loss':['linear','square','exponential'],
                     'n_estimators':[200,250,300,350,400,450],
                     'learning_rate':[0.05,0.1,0.25,0.5,0.75,1]}

# Two considered error scores 
scoresMSE = make_scorer(mean_squared_error)
scores = [scoresMSE,'r2']
scores_Name = ["MSE","r2"]

for N_Pro in range(7): 
    # Load data
    Title_file = file_list[N_Pro]
    with open("Resume_ML_FreeSurface_%s.txt" %Title_file,"a") as f:
        xArr,yArr = loadDataSet("FCC_FreeSurface_%s.txt" % Title_file)
        X = np.array(xArr)
        y = np.array(yArr)
        Label_xy = ["Measured values of %s" % Label_list[N_Pro],"Predicted values of %s" % Label_list[N_Pro]]
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        # Only store the scores of the feature importance 
        with open("Resume_ImFac_%s.txt" %Title_file,"a") as fIF:
            f.write("\n \n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            f.write("Results of %s \n" % Label_list[N_Pro])
            fIF.write("Results of feature importances analyses for %s \n" % Label_list[N_Pro])
            fIF.write(str(['G1','G2','k','A','a','SFE','CHE','theta','phi']))
            fIF.write('\n')
            for N_testing_set in range(5):
                f.write("\n****************************\n")
                offset1 = int(X.shape[0] * Test_set_list[N_testing_set])
                offset2 = int(X.shape[0] * Test_set_list[N_testing_set + 1])
                X_train1, y_train1 = X[:offset1,:], y[:offset1]
                X_train2, y_train2 = X[offset2:,:], y[offset2:]
                X_train = np.concatenate((X_train1,X_train2),axis=0)
                y_train = np.concatenate((y_train1,y_train2),axis=0)
                X_test, y_test = X[offset1:offset2,:], y[offset1:offset2]
                for i in range(2):
                    f.write("----------------------------\n")
                    f.write("# Tuning hyper-parameters of %s with testing set %s - %s by %s \n" 
                            %(Label_list[N_Pro],Name_Test_List[N_testing_set],Name_Test_List[N_testing_set + 1],scores_Name[i]))

                    # Use a grid-search optimization strategy to find the best hyper-parameters 
                    clf = GridSearchCV(clf_DTR,tuned_parameters,cv=5,scoring=scores[i])
                    clf.fit(X_train,y_train)

                    # Save the best hyper-parameters 
                    f.write("Best parameters set found on development set: \n")
                    for item in clf.best_params_.items():
                        f.write("%s: %s \n" %(item[0],str(item[1])))
    
                    # Choose the best hyper-parameters from the GridSearch and use them to train a model 
                    clf_Best = clf.best_estimator_
                    clf_Best.fit(X_train,y_train)

                    # Save the model with the best hyper-parameters
                    joblib.dump(clf_Best,"./clf_%s_%s - %s_%s.pkl" %(Title_file,Name_Test_List[N_testing_set],Name_Test_List[N_testing_set + 1],scores_Name[i]),compress=3)
                    # Use the following function to load the exist model
                    # clf_Best=joblib.load("Name_of_exist_model.pkl")
                    
                    # Predict the results for the test dataset
                    y_pred = clf_Best.predict(X_test)
                    # Determine the error scores between the predictions and the calculated results for the test dataset 
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    # Save the error scores
                    f.write("MSE of test: %.8f \n" % mse)
                    f.write("r2 of test: %.8f \n" % r2)
                    
                    # Plot and save the Figure of predictions compared to calculated results for the test dataset
                    y_all = np.concatenate((y_test, y_pred),axis=0)
                    data_min = y_all.min()
                    data_max = y_all.max()
                    cri_min = data_min - (data_max - data_min) * 0.1
                    cri_max = data_max + (data_max - data_min) * 0.1
                    plt.plot([cri_min,cri_max],[cri_min,cri_max],color='b')
                    plt.scatter(y_test, y_pred, s=60, color="r", linewidths=1, marker='o', edgecolors='k')
                    plt.xlim(cri_min,cri_max)
                    plt.ylim(cri_min,cri_max)
                    plt.xlabel(Label_xy[0])
                    plt.ylabel(Label_xy[1])
                    plt.savefig("./%s_%s - %s_%s_Pre.png" %(Title_file,Name_Test_List[N_testing_set],Name_Test_List[N_testing_set + 1],scores_Name[i]))
                    plt.show()

                    # Save the input factors, calculated results and the predictions for the test dataset 
                    Data_M_P = np.concatenate((X_test, y_test[:,np.newaxis], y_pred[:,np.newaxis]),axis=1)
                    np.savetxt("./%s_%s - %s_%s_Data_Measure_Predict.txt" %(Title_file,Name_Test_List[N_testing_set],Name_Test_List[N_testing_set + 1],scores_Name[i]), Data_M_P, fmt="%.14f", delimiter=' ')

                    feature_importance_df = pd.DataFrame({
                        'name':['G1','G2','k','A','a','SFE','CHE','theta','phi'],
                        'importance':clf_Best.feature_importances_})

                    # Save the feature importance for considered features 
                    IFoutput=str(clf_Best.feature_importances_)
                    f.write('Feature importances: \n')
                    f.write(str(['G1','G2','k','A','a','SFE','CHE','theta','phi']))
                    f.write('\n')
                    f.write(IFoutput)
                    f.write('\n')
                    fIF.write(IFoutput)
                    fIF.write('\n')

                    # Plot and save the Figure of feature importance analyses 
                    feature_importance_df.sort_values(by = 'importance',ascending = False,inplace = True)
                    x_axis = list(feature_importance_df['name'])
                    y_axis = list(feature_importance_df['importance'])
                    plt.title('Importance of Factors')
                    plt.bar(x_axis,y_axis)
                    plt.savefig("./%s_%s - %s_%s_FI.png" %(Title_file,Name_Test_List[N_testing_set],Name_Test_List[N_testing_set + 1],scores_Name[i]))
                    plt.show()
            fIF.close()
        f.close()
print("Done")