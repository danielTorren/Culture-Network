"""ANN sensitivity analysis, should probs be a jupyter note book tbh
[COMPLETE]

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import json
import numpy as np
import numpy.typing as npt
from resources.SA_sobol import (
    produce_param_list_SA,
    get_plot_data,  
    Merge_dict_SA,
    analyze_results,  
)
from resources.plot import (
    prints_SA_matrix,
    bar_sensitivity_analysis_plot,
    scatter_total_sensitivity_analysis_plot,
    multi_scatter_total_sensitivity_analysis_plot,
    multi_scatter_sidebyside_total_sensitivity_analysis_plot,
    multi_scatter_seperate_total_sensitivity_analysis_plot,
    multi_scatter_parallel_total_sensitivity_analysis_plot,
)
from resources.utility import (
    load_object,
    save_object,
    createFolder,
)
from SALib.sample import saltelli
from sklearn.model_selection import train_test_split,GridSearchCV,cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.decomposition import PCA

def produce_param_list_ANN_SA(
    param_values: npt.NDArray, base_params: dict, variable_parameters_dict: dict[dict]
) -> list:
    """
    Generate the list of dictionaries containing informaton for each experiment. We combine the base_params with the specific variation for
    that experiment from param_values and we just use variable_parameters_dict for the property

    Parameters
    ----------
    param_values: npt.NDArray
        the set of parameter values which are tested in the sensitivity analysis, generated using saltelli.sample - see:
        https://salib.readthedocs.io/en/latest/api/SALib.sample.html#SALib.sample.saltelli.sample for more details
    base_params: dict
        This is the set of base parameters which act as the default if a given variable is not tested in the sensitivity analysis.
        See sa_run for example data structure
    variable_parameters_dict: dict[dict]
        These are the parameters to be varied. The data is structured as follows: The key is the name of the property, the value is composed
        of another dictionary which contains itself several properties; "property": the parameter name, "min": the minimum value to be used in
        the sensitivity analysis, "max": the maximum value to be used in the sensitivity analysis, "title": the name to be used when plotting
        sensitivity analysis results. See sa_run for example data structure.

    Returns
    -------
    params_list: list[dict]
        list of parameter dicts, each entry corresponds to one experiment to be tested
    """

    params_list = []
    for i, X in enumerate(param_values):
        base_params_copy = (
            base_params.copy()
        )  # copy it as we dont want the changes from one experiment influencing another

        variable_parameters_dict_toList = list(
            variable_parameters_dict.values()
        )  # turn it too a list so we can loop through it as X is just an array not a dict

        for v in range(len(X)):  # loop through the properties to be changed
            base_params_copy[variable_parameters_dict_toList[v]["property"]] = X[
                v
            ]  # replace the base variable value with the new value for that experiment
        params_list.append(base_params_copy)

    return params_list

def mlp_model_select(X, Y,param_grid,estimator):
    """taken from a stack overflow question"""

    """
    best_params {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': 700, 'learning_rate': 'adaptive', 'max_iter': 100, 'solver': 'adam'}
    """
    gsc = GridSearchCV(
        estimator,
        param_grid,
        cv=5,#cross validation folds 
        scoring='neg_mean_squared_error', 
        verbose=1, 
        n_jobs=-1,#run jobs in parallel
        )

    grid_result = gsc.fit(X, Y)
    #print("grid results",grid_result)

    best_params = grid_result.best_params_
    print("best_params",best_params)

    best_mlp = MLPRegressor(hidden_layer_sizes = best_params["hidden_layer_sizes"], 
                            activation =best_params["activation"],
                            alpha = best_params['alpha'],
                            learning_rate = best_params['learning_rate'],
                            solver=best_params["solver"],
                            max_iter = best_params['max_iter'],
                )

    scoring = ['r2', 'neg_mean_squared_error','neg_mean_absolute_error']

    scores = cross_validate(best_mlp, X, Y, cv=10, scoring=scoring, return_train_score=True, return_estimator = True)
    return scores, best_mlp ,gsc

def generate_problem_ANN(
    variable_parameters_dict: dict[dict],
    N_samples: int,
    calc_second_order: bool,
) -> tuple[dict, str, npt.NDArray]:
    """
    Generate the saltelli.sample given an input set of base and variable parameters, generate filename and folder. Satelli sample used
    is 'a popular quasi-random low-discrepancy sequence used to generate uniform samples of parameter space.' - see the SALib documentation

    Parameters
    ----------
    variable_parameters_dict: dict[dict]
        These are the parameters to be varied. The data is structured as follows: The key is the name of the property, the value is composed
        of another dictionary which contains itself several properties; "property": the parameter name, "min": the minimum value to be used in
        the sensitivity analysis, "max": the maximum value to be used in the sensitivity analysis, "title": the name to be used when plotting
        sensitivity analysis results. See sa_run for example data structure.
    N_samples: int
        Number of samples taken per parameter, If calc_second_order is False, the Satelli sample give N * (D + 2), (where D is the number of parameter) parameter sets to run the model
        .There are then extra runs per parameter set to account for stochastic variation. If calc_second_order is True, then this is N * (2D + 2) parameter sets.
    AV_reps: int
        number of repetitions performed to average over stochastic effects
    calc_second_order: bool
        Whether or not to conduct second order sobol sensitivity analysis, if set to False then only first and total order results will be
        available. Setting to True increases the total number of runs for the sensitivity analysis but allows for the study of interdependancies
        between parameters
    Returns
    -------
    problem: dict
        Outlines the number of variables to be varied, the names of these variables and the bounds that they take
    fileName: str
        name of file where results may be found
    param_values: npt.NDArray
        the set of parameter values which are tested in the sensitivity analysis, generated using saltelli.sample - see:
        https://salib.readthedocs.io/en/latest/api/SALib.sample.html#SALib.sample.saltelli.sample for more details
    """
    D_vars = len(variable_parameters_dict)
    
    if calc_second_order:
        samples = N_samples * (2*D_vars + 2)
    else:
        samples = N_samples * (D_vars + 2)

    print("samples: ", samples)

    names_list = [x["property"] for x in variable_parameters_dict.values()]
    bounds_list = [[x["min"], x["max"]] for x in variable_parameters_dict.values()]
    round_variable_list = [x["property"] for x in variable_parameters_dict.values() if x["round"]]

    problem = {
        "num_vars": D_vars,
        "names": names_list,
        "bounds": bounds_list,
    }

    ########################################
    fileName = "results/ANN_samples_%s_D_vars_%s_N_samples_%s" % (
        str(samples),
        str(D_vars),
        str(N_samples),
    )
    print("fileName: ", fileName)
    createFolder(fileName)

    # GENERATE PARAMETER VALUES
    param_values = saltelli.sample(
        problem, N_samples, calc_second_order=calc_second_order
    )  # NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters

    print("round_variable_list:",round_variable_list)
    for i in round_variable_list:
        index_round = problem["names"].index(i)
        param_values[:,index_round] = np.round(param_values[:,index_round])

    return problem, fileName, param_values

def sa_run_ANN(
    N_samples: int,
    variable_parameters_dict: dict,
    calc_second_order: bool,
    emulator
) -> tuple[int, dict, str, dict, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Generate the list of dictionaries of parameters for different simulation runs, run them and then save the data from the runs.
    """

    ##AVERAGE RUNS
    problem, fileName, param_values = generate_problem_ANN(
        variable_parameters_dict, N_samples,  calc_second_order
    )

    scaler = MinMaxScaler()  
    # Don't cheat - fit only on training data
    scaler.fit(param_values)  
    X_scaled = scaler.transform(param_values)  

    y_pred_mlp_reg = emulator.predict(X_scaled)

    Y_emissions = y_pred_mlp_reg[:,0]
    Y_mu = y_pred_mlp_reg[:,1] 
    Y_var = y_pred_mlp_reg[:,2] 
    Y_coefficient_of_variance = y_pred_mlp_reg[:,3] 
    Y_emissions_change = y_pred_mlp_reg[:,4]

    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
    save_object(problem, fileName + "/Data", "problem")
    save_object(Y_emissions, fileName + "/Data", "Y_emissions")
    save_object(Y_mu, fileName + "/Data", "Y_mu")
    save_object(Y_var, fileName + "/Data", "Y_var")
    save_object(Y_coefficient_of_variance, fileName + "/Data", "Y_coefficient_of_variance")
    save_object(Y_emissions_change, fileName + "/Data", "Y_emissions_change")

    return (
        problem,
        fileName,
        param_values,
        Y_emissions,
        Y_mu,
        Y_var,
        Y_coefficient_of_variance,
        Y_emissions_change
    )


# constants
dpi_save = 1200

plot_dict = {
    "emissions": {"title": r"$E/NM$", "colour": "red", "linestyle": "--"},
    "mu": {"title": r"$\mu$", "colour": "blue", "linestyle": "-"},
    "var": {"title": r"$\sigma^{2}$", "colour": "green", "linestyle": "*"},
    "coefficient_of_variance": {"title": r"$\sigma/\mu$","colour": "orange","linestyle": "-.",},
    "emissions_change": {"title": r"$\Delta E/NM$", "colour": "brown", "linestyle": "-*"},
}

titles = [
    r"Number of individuals, $N$", 
    r"Number of behaviours, $M$", 
    r"Mean neighbours, $K$",
    r"Probability of re-wiring, $p_r$",
    r"Cultural momentum, $\rho$",
    r"Social learning error, $\varepsilon$",
    r"Attitude Beta $a$",
    r"Attitude Beta $b$",
    r"Threshold Beta $a$",
    r"Threshold Beta $b$",
    r"Discount factor, $\delta$",
    r"Attribute homophily, $h$",
    r"Confirmation bias, $\theta$"
]

"""
THIS WAS USE FOR THE LOADED RUN, JUST USEFUL AS A REFERENCEgit 
{
    "N":{"property": "N","min":100,"max":300, "title": "$N$"},
    "M":{"property":"M","min":1,"max": 30, "title": "$M$"},
    "K":{"property":"K","min":5,"max":80 , "title": "$K$"},
    "prob_rewire": {"property": "prob_rewire","min": 0.0,"max": 1.0,"title": "$p_r$"},
    "culture_momentum_real":{"property":"culture_momentum_real","min":1,"max": 2500, "title": "$T_{\\rho}$"},
    "learning_error_scale":{"property":"learning_error_scale","min":0.0,"max":0.8 , "title": "$\\epsilon$" },
    "a_attitude":{"property":"a_attitude","min":0.05, "max": 8, "title": "$a$ Attitude"},
    "b_attitude":{"property":"b_attitude","min":0.05, "max":8 , "title": "$b$ Attitude"},
    "a_threshold":{"property":"a_threshold","min":0.05, "max": 8, "title": "$a$ Threshold"},
    "b_threshold":{"property":"b_threshold","min":0.05, "max": 8, "title": "$b$ Threshold"},
    "discount_factor":{"property":"discount_factor","min":0.0, "max":1.0 , "title": "$\\delta$"},
    "homophily": {"property": "homophily","min": 0.0,"max": 1.0,"title": "$h$"},
    "confirmation_bias":{"property":"confirmation_bias","min":-10.0, "max":100 , "title": "$\\theta$"}
}
"""

#THIS IS FOR CREATING THE ANN MODEL FROM SENSITVITY ANALYSIS DATA
generate_model = 1
#where to get the data to train the model on
fileNameModel = "results\SA_AV_reps_5_samples_15360_D_vars_13_N_samples_1024"#\SA_AV_reps_5_samples_28672_D_vars_13_N_samples_1024"#
N_samplesModel = 256#512#64#1024#512
calc_second_orderModel = False
hyper_parameters_fit = 0

#THIS IS FOR GENERATING THE DATA FROM THE ANN TO RUN SENSITIVITY ANALYSIS
RUN_sensitvity = 0
N_samplesSA = 2048#512#64#1024#512
calc_second_orderSA = False


if __name__ == "__main__":
    if generate_model:
        #load datt
        variable_parameters_dict = load_object(fileNameModel + "/Data", "variable_parameters_dict")

        problem = load_object(fileNameModel + "/Data", "problem")
        
        #################################################################################################################
        """REAPEATED CODE"""
        # GENERATE PARAMETER VALUES
        X = saltelli.sample(
            problem, N_samplesModel, calc_second_order=calc_second_orderModel
        )  # NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters

        """
        round_variable_list = [x["property"] for x in variable_parameters_dict.values() if x["round"]]
        print("round_variable_list:",round_variable_list)
        for i in round_variable_list:
            index_round = problem["names"].index(i)
            X[:,index_round] = np.round(X[:,index_round])
        """
        ######################################################################################################

        Y_emissions = load_object(fileNameModel+ "/Data", "Y_emissions")
        Y_mu = load_object(fileNameModel + "/Data", "Y_mu")
        Y_var = load_object(fileNameModel + "/Data", "Y_var")
        Y_coefficient_of_variance = load_object(fileNameModel + "/Data", "Y_coefficient_of_variance")
        Y_emissions_change = load_object(fileNameModel + "/Data", "Y_emissions_change")

        output_lists = [Y_emissions ,Y_mu, Y_var, Y_coefficient_of_variance, Y_emissions_change]#Y_emissions,Y_mu, Y_var, Y_coefficient_of_variance,Y_emissions_change

        #get out the inputs and results as correct format
        print("Shape inputs", X.shape)
        y = np.asarray([[x[i] for x in output_lists] for i in range(len(output_lists[0]))])
        print("Shape outputs", y[0],y.shape)
        
        
        #split data
        X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
        print("train and test sizes", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        #scale the data, I think you just do it on the features?
        #scaler = StandardScaler()  
        scaler = MinMaxScaler()  
        # Don't cheat - fit only on training data
        scaler.fit(X_train)  
        X_train_scaled = scaler.transform(X_train)  
        # apply same transformation to test data
        X_test_scaled = scaler.transform(X_test) 

        #scaler_y  = StandardScaler()  
        scaler_y = MinMaxScaler()  
        # Don't cheat - fit only on training data
        scaler_y.fit(y_train)  
        y_train_scaled = scaler_y.transform(y_train)  
        # apply same transformation to test data
        y_test_scaled = scaler_y.transform(y_test) 

        if hyper_parameters_fit:
            #need to fine tune the hyper parameters
            estimator=MLPRegressor()#LinearRegression()

            """NO IDEA IF THESE ARE GOOD PARAMS TO TEST"""
            param_grid = {
                    'hidden_layer_sizes': [10,50,100, 200, 500],
                    'activation': ['relu','tanh','logistic'],#'activation': ['relu','tanh','logistic'],
                    'alpha': [0.001,0.01, 0.05],#'alpha': [0.0001, 0.05],,0.001,0.01, 0.05
                    'learning_rate': ['adaptive'],
                    'solver': ['adam'],
                    'max_iter': [100,200,300,400,500],#[200,500]
                    }
            scores, mlp_reg, gsc= mlp_model_select(X_train_scaled, y_train_scaled,param_grid ,estimator)
            #print("scores",scores)
            #print("best mlp",mlp_reg)

            #print("gsc.cv_results_",gsc.cv_results_['mean_test_score'])

            """
            x_axis= param_grid['hidden_layer_sizes']
            fig3, ax3 = plt.subplots()
            ax3.plot(x_axis,-gsc.cv_results_['mean_test_score'], label='Test Error')
            ax3.plot(x_axis,-gsc.cv_results_['mean_train_score'], label='Train Error')
            ax3.legend()
            ax3.set_xlabel('Number of neurons')
            ax3.set_ylabel('Objective function value')
            """
        else:
            mlp_reg = LinearRegression()
            """
            mlp_reg = MLPRegressor(
                hidden_layer_sizes = 500,
                activation = 'relu',
                alpha = 0.05,
                learning_rate='adaptive',
                solver = 'adam',
                max_iter = 300
            )
            """

        ###best_params {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': 500, 'learning_rate': 'adaptive', 'max_iter': 400, 'solver': 'adam'}
        print("mlp ready")


        """
        Im getting half decent results on the larger sensitivity run, ~ 28000 data points,  but not great on the smaller one ~ 3000
        """
        

        ##############################################################
        #run the MPL
        
        mlp_reg.fit(X_train_scaled, y_train_scaled)
        print("fitted")


        ###SAVE THE MODEL
        save_object(mlp_reg, fileNameModel + "/Data", "mlp_reg")

        y_pred_mlp_reg = mlp_reg.predict(X_test_scaled)
        print("predicted")

        print("R^2 : ", r2_score(y_test_scaled, y_pred_mlp_reg))
        print("MAE :", mean_absolute_error(y_test_scaled,y_pred_mlp_reg))
        print("RMSE:",np.sqrt(mean_squared_error(y_test_scaled, y_pred_mlp_reg)))

        #score is coefficient of determination of the prediction
        #he best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse)
        #NOT GOOD
        #train_score = mlp_reg.score(X_train_scaled, y_train)
        #print("train_score (coefficient of determination)",train_score)
        #test_score = mlp_reg.score(X_test_scaled, y_test)
        #print("test_score (coefficient of determination)", test_score)

        
        #Loss Curve
        """
        fig1, ax1 = plt.subplots()
        ax1.plot(mlp_reg.loss_curve_)
        ax1.set_title("Loss Curve")
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Cost')
        """

        #############################################################################################
        PCA_plots = 0
        if PCA_plots:
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
                    print("pca.explained_variance_ratio_",pca.explained_variance_ratio_)
                    print("pca.singular_values_",pca.singular_values_)

                    axes[i].plot(np.cumsum(pca.explained_variance_ratio_))

        

        ##########################################################
        """Try gaussian process regressor
        gpr = GaussianProcessRegressor()

        gpr.fit(X_train_scaled, y_train_scaled.ravel())
        print("fitted gpr")

        y_pred_gpr = gpr.predict(X_test_scaled)
        print("predicted gpr")

        print("R^2 gpr: ", r2_score(y_test_scaled, y_pred_gpr))
        print("MAE gpr:", mean_absolute_error(y_test_scaled,y_pred_gpr))
        print("RMSE gpr:",np.sqrt(mean_squared_error(y_test_scaled, y_pred_gpr)))
        """
    if RUN_sensitvity:
        # load variable params
        f_variable_parameters = open(
            "constants/variable_parameters_dict_SA.json"
        )
        variable_parameters_dict = json.load(f_variable_parameters)
        f_variable_parameters.close()

        emmulator = load_object(fileNameModel + "/Data", "mlp_reg")

        (
            problem,
            fileName,
            param_values,
            Y_emissions,
            Y_var,
            Y_mu,
            Y_coefficient_of_variance,
            Y_emissions_change,
        ) = sa_run_ANN(N_samplesSA, variable_parameters_dict, calc_second_orderSA, emmulator)
        

        data_sa_dict_total, data_sa_dict_first = get_plot_data(
            problem, Y_emissions, Y_mu, Y_var, Y_coefficient_of_variance,Y_emissions_change, calc_second_orderSA
        )#here is where mu and var were the worng way round!

        #print([x["title"] for x in variable_parameters_dict.values()])
        #print("titles check order",titles)
        #quit()

        data_sa_dict_total = Merge_dict_SA(data_sa_dict_total, plot_dict)
        data_sa_dict_first = Merge_dict_SA(data_sa_dict_first, plot_dict)

        ######
        # PLOTS - COMMENT OUT THE ONES YOU DONT WANT

        #data_sa_dict_first[""]
        # multi_scatter_total_sensitivity_analysis_plot(fileName, data_sa_dict_total,titles, dpi_save,N_samples, "Total")
        # multi_scatter_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, titles, dpi_save,N_samples, "First")
        #multi_scatter_sidebyside_total_sensitivity_analysis_plot(fileName, data_sa_dict_total, titles, dpi_save,N_samples, "Total")
        #multi_scatter_sidebyside_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, titles, dpi_save,N_samples, "First")
        multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, titles, dpi_save, N_samplesSA, "First")
        #multi_scatter_parallel_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, titles, dpi_save, N_samples, "First")
        #multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_total, titles, dpi_save, N_samples, "Total")

    plt.show()
