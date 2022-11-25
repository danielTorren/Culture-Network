"""ANN sensitivity analysis, should probs be a jupyter note book tbh
[COMPLETE]

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
from resources.utility import (
    load_object,
)
from resources.SA_sobol import (
    generate_problem,
    produce_param_list_SA
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.decomposition import PCA

# constants
fileName = "results\SA_AV_reps_5_samples_3840_D_vars_13_N_samples_256"
N_samples = 256#512#64#1024#512
calc_second_order = False
# Visualize
dpi_save = 1200

if __name__ == "__main__":
    ######FOR FIRST RUN NEED TO LOAD IN BASE AS I FOR GOT TO SAVE THE LIST OF PARAMS THE ONLY ONES THAT MATTER AR AV REPS AN PHI AND Total time
    # load base params
    f = open("constants/base_params.json")
    base_params = json.load(f)
    base_params["time_steps_max"] = int(
        base_params["total_time"] / base_params["delta_t"]
    )

    AV_reps = len(base_params["seed_list"])

    #load datt
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")

    problem = load_object(fileName + "/Data", "problem")

    problem, fileName, param_values = generate_problem(
        variable_parameters_dict, N_samples, AV_reps, calc_second_order
    )

    params_list_sa = produce_param_list_SA(
        param_values, base_params, variable_parameters_dict
    )

    Y_emissions = load_object(fileName + "/Data", "Y_emissions")
    Y_mu = load_object(fileName + "/Data", "Y_mu")
    Y_var = load_object(fileName + "/Data", "Y_var")
    Y_coefficient_of_variance = load_object(
        fileName + "/Data", "Y_coefficient_of_variance"
    )
    Y_emissions_change = load_object(fileName + "/Data", "Y_emissions_change")

    output_lists = [Y_var]#Y_emissions,Y_mu, Y_var, Y_coefficient_of_variance,Y_emissions_change


    #get out the inputs and results as correct format
    X = np.asarray([[v[x] for x in v.keys() if x in variable_parameters_dict.keys()] for v in params_list_sa])
    #print("Shape inputs", X.shape)
    y = np.asarray([[x[i] for x in output_lists] for i in range(len(output_lists[0]))])
    #print("Shape outputs", y[0],y.shape)
    

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
    print("train and test sizes", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    #scale the data, I think you just do it on the features?
    scaler = StandardScaler()  
    # Don't cheat - fit only on training data
    scaler.fit(X_train)  
    X_train_scaled = scaler.transform(X_train)  
    # apply same transformation to test data
    X_test_scaled = scaler.transform(X_test) 

    #need to fine tune the hyper parameters
    
    #run the MPL
    mlp_reg = MLPRegressor(max_iter = 500)
    print("mlp ready")
    mlp_reg.fit(X_train_scaled, y_train)
    print("fitted")
    y_pred = mlp_reg.predict(X_test_scaled)
    print("predicted")

    print("R^2 : ", r2_score(y_test, y_pred))
    print("MAE :", mean_absolute_error(y_test,y_pred))
    print("RMSE:",np.sqrt(mean_squared_error(y_test, y_pred)))

    #score is coefficient of determination of the prediction
    #he best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse)
    #NOT GOOD
    train_score = mlp_reg.score(X_train_scaled, y_train)
    print("train_score (coefficient of determination)",train_score)
    test_score = mlp_reg.score(X_test_scaled, y_test)
    print("test_score (coefficient of determination)", test_score)

    #Loss Curve
    fig1, ax1 = plt.subplots()
    ax1.plot(mlp_reg.loss_curve_)
    ax1.set_title("Loss Curve")
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cost')

    #run the PCA, NEED TO GENERATE NEW X DATA AND INTERPOLATE USING THE MODEL!
    X_PCA = scaler.transform(X)#X_test_scaled
    y_PCA = mlp_reg.predict(X_PCA)#y_test
    
    if len(output_lists) == 1.0:
        pca = PCA()
        pca.fit(X_PCA,y_PCA)
        fig2, ax2 = plt.subplots()
        ax2.plot(np.cumsum(pca.explained_variance_ratio_))
        ax2.set_title("Choosing the number of components")
        ax2.set_xlabel('number of components')
        ax2.set_ylabel('cumulative explained variance')
    else:
        fig, axes = plt.subplots(len(output_lists),1, sharex=True)
        fig.suptitle("Choosing the number of components")
        fig.supxlabel('number of components')
        fig.supylabel('cumulative explained variance')

        for i in range(len(output_lists)):
            pca = PCA()
            pca.fit(X_PCA,y_PCA[:,i])
            #print("pca.explained_variance_ratio_",pca.explained_variance_ratio_)
            #print("pca.singular_values_",pca.singular_values_)

            axes[i].plot(np.cumsum(pca.explained_variance_ratio_))

    plt.show()
