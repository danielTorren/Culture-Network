from utility import (
    sa_save_problem,
    sa_save_Y,
    sa_load_problem,
    sa_load_Y,
    createFolderSA,
    produce_param_list_SA
)
from run import parallel_run_sa, average_seed_parallel_run_sa
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import random
from datetime import datetime
import matplotlib.pyplot as plt
from plot import (
    prints_SA_matrix,
    bar_sensitivity_analysis_plot,
)
#print("Hey",random.seed(datetime.now()))

base_params = {
    "seed_list": [1,2,3,4,5],
    "total_time": 100,
    "delta_t": 0.05,
    "compression_factor": 10,
    "save_data": True, 
    "alpha_change" : 1.0,
    "harsh_data": False,
    "averaging_method": "Arithmetic",
    "phi_list_lower": 0.1,
    "phi_list_upper": 1.0,
    "N": 50,
    "M": 3,
    "K": 20,
    "prob_rewire": 0.05,
    "set_seed": 0, #SET SEED TO 0 IF YOU WANT RANDOM SEED! #1,#None,#random.seed(datetime.now()),#1,
    "culture_momentum_real": 5,
    "learning_error_scale": 0.02,
    "discount_factor": 0.6,
    "present_discount_factor": 0.8,
    "inverse_homophily": 0.1,#1 is total mixing, 0 is no mixing
    "homophilly_rate" : 1.5,
    "confirmation_bias": 25,
}

base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

#behaviours!
if base_params["harsh_data"]:#trying to create a polarised society!
    base_params["green_extreme_max"]= 8
    base_params["green_extreme_min"]= 2
    base_params["green_extreme_prop"]= 2/5
    base_params["indifferent_max"]= 2
    base_params["indifferent_min"]= 2
    base_params["indifferent_prop"]= 1/5
    base_params["brown_extreme_min"]= 2
    base_params["brown_extreme_max"]= 8
    base_params["brown_extreme_prop"]= 2/5
    if base_params["green_extreme_prop"] + base_params["indifferent_prop"] + base_params["brown_extreme_prop"] != 1:
        raise Exception("Invalid proportions")
else:
    base_params["alpha_attitude"] = 0.1
    base_params["beta_attitude"] = 0.1
    base_params["alpha_threshold"] = 2
    base_params["beta_threshold"] = 2

#################################

# variable parameters
"""THE ORDER MATTERS WHEN REPLOTING, FIX THIS!"""
variable_parameters_dict = [
    {"property": "N","min":50,"max":200, "title": r"$N$"}, 
    {"property":"M","min":1,"max": 10, "title": r"$M$"}, 
    {"property":"K","min":2,"max":30 , "title": r"$K$"}, 
    {"property":"prob_rewire","min":0.0, "max":0.2 , "title": r"$p_r$"}, 
    #{"property":"set_seed","min":1, "max":10000, "title": r"Seed"}, 
    {"property":"culture_momentum_real","min":1,"max": 50, "title": r"$T_{\rho}$"}, 
    {"property":"learning_error_scale","min":0.0,"max":0.5 , "title": r"$\eta$" }, 
    #{"property":"alpha_attitude","min":0.1, "max": 8, "title": r"Attitude $\alpha$"}, 
    #{"property":"beta_attitude","min":0.1, "max":8 , "title": r"Attitude $\beta$"}, 
    #{"property":"alpha_threshold","min":0.1, "max": 8, "title": r"Threshold $\alpha$"}, 
    #{"property":"beta_threshold","min":0.1, "max": 8, "title": r"Threshold $\beta$"}, 
    {"property":"discount_factor","min":0.0, "max":1.0 , "title": r"$\delta$"}, 
    {"property":"inverse_homophily","min":0.0, "max": 1.0, "title": r"$h$"}, 
    {"property":"present_discount_factor","min":0.0, "max": 1.0, "title": r"$\beta$"}, 
    {"property":"confirmation_bias","min":0.0, "max":100 , "title": r"$\theta$"}, 
]


# Visualize
dpi_save = 1200

def generate_problem(variable_parameters_dict,N_samples):

    D_vars = len(variable_parameters_dict)
    samples = N_samples * (2*D_vars + 2)
    print("samples: ",samples)

    names_list = [x["property"] for x in variable_parameters_dict]
    bounds_list = [[x["min"],x["max"]] for x in variable_parameters_dict]

    problem = {
        "num_vars": D_vars,
        "names": names_list,
        "bounds": bounds_list,
    }

    ########################################

    fileName = "results/SA_%s_%s_%s" % (str(samples),str(D_vars),str(N_samples))
    print("fileName: ", fileName)
    createFolderSA(fileName)
    
    # GENERATE PARAMETER VALUES
    param_values = saltelli.sample(
        problem, N_samples
    )  # NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters

    return problem, fileName, param_values

def average_generate_problem(variable_parameters_dict,N_samples,average_reps):

    D_vars = len(variable_parameters_dict)
    samples = N_samples * (2*D_vars + 2)
    print("samples: ",samples)

    names_list = [x["property"] for x in variable_parameters_dict]
    bounds_list = [[x["min"],x["max"]] for x in variable_parameters_dict]

    problem = {
        "num_vars": D_vars,
        "names": names_list,
        "bounds": bounds_list,
    }

    ########################################

    fileName = "results/average_SA_%s_%s_%s_%s" % (str(average_reps),str(samples),str(D_vars),str(N_samples))
    print("fileName: ", fileName)
    createFolderSA(fileName)
    
    # GENERATE PARAMETER VALUES
    param_values = saltelli.sample(
        problem, N_samples
    )  # NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters

    return problem, fileName, param_values

def generate_sa_data(base_params,variable_parameters_dict,param_values,results_property):

    params_list_sa = produce_param_list_SA(param_values,base_params,variable_parameters_dict)

    Y = parallel_run_sa(params_list_sa,results_property)
    
    return np.asarray(Y)

def average_generate_sa_data(base_params,variable_parameters_dict,param_values,results_property):

    params_list_sa = produce_param_list_SA(param_values,base_params,variable_parameters_dict)

    Y = average_seed_parallel_run_sa(params_list_sa,results_property)
    
    return np.asarray(Y)

def get_data_bar_chart(Si_df):
    """
        Parameters
    ----------
    * Si_df: pd.DataFrame, of sensitivity results

    TAKEN FROM THE SALIB BAR PLOT: https://salib.readthedocs.io/en/latest/_modules/SALib/plotting/bar.html

    """
    # magic string indicating DF columns holding conf bound values
    conf_cols = Si_df.columns.str.contains('_conf')
    confs = Si_df.loc[:, conf_cols]#select all those that ARE in conf_cols!
    confs.columns = [c.replace('_conf', '') for c in confs.columns]
    Sis = Si_df.loc[:, ~conf_cols]#select all those that ARENT in conf_cols!

    return Sis,confs

if __name__ == "__main__":

    RUN = 0#False,True


    if RUN:
        results_property = "Carbon Emissions/NM"
        N_samples = 4

        """
        # run the thing
        problem, fileName, param_values = generate_problem(variable_parameters_dict,N_samples)
        sa_save_problem(problem,fileName)
        Y = generate_sa_data(base_params,variable_parameters_dict,param_values,results_property)
        sa_save_Y(Y,fileName)
        """

        ##AVERAGE RUNS
        average_reps = len(base_params["seed_list"])
        print(average_reps)
        average_problem, average_fileName, average_param_values = average_generate_problem(variable_parameters_dict,N_samples,average_reps)
        sa_save_problem(average_problem,average_fileName)
        average_Y = average_generate_sa_data(base_params,variable_parameters_dict,average_param_values,results_property)
        sa_save_Y(average_Y,average_fileName)
    else:
        average_fileName = "results/average_SA_5_88_10_4"
        N_samples = 4
        average_problem = sa_load_problem(average_fileName)
        average_Y = sa_load_Y(average_fileName)
    
    #Si = sobol.analyze(problem, Y, print_to_console=False)
    Si = sobol.analyze(average_problem, average_Y, print_to_console=False)

    ###PLOT RESULTS
    
    names = [x["title"] for x in variable_parameters_dict]

    #### Bar chart
    total, first, second = Si.to_df()
    data_sa, yerr = get_data_bar_chart(total)
    #bar_sensitivity_analysis_plot(fileName, data_sa, names, yerr , dpi_save,N_samples)
    bar_sensitivity_analysis_plot(average_fileName, data_sa, names, yerr , dpi_save,N_samples)

    #Matrix plot
    data = [np.asarray(Si["S2"]),np.asarray(Si["S2_conf"])]
    title_list = ["S2","S2_conf"]
    cmap = "Blues"
    nrows = 1
    ncols = 2
    prints_SA_matrix(average_fileName, data,title_list,cmap,nrows, ncols, dpi_save, names,N_samples)

    plt.show()