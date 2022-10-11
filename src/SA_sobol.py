"""Performs sobol sensitivity analysis on the model. 
A module that constructs dictionary of data to be varied then runs the multiple repeated simulations, 
to account for stochastic variability. Then saves data and parameters used to produce that sensitivity 
analysis data. Finally plots the results.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

#imports
from utility import (
    sa_save_problem,
    sa_save_Y,
    sa_load_problem,
    sa_load_Y,
    createFolderSA,
    add_varaiables_to_dict
)
from run import parallel_run_sa
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
from plot import (
    prints_SA_matrix,
    bar_sensitivity_analysis_plot,
    scatter_total_sensitivity_analysis_plot,
    multi_scatter_total_sensitivity_analysis_plot,
    multi_scatter_sidebyside_total_sensitivity_analysis_plot,
    multi_scatter_separate_total_sensitivity_analysis_plot
)

#constants
base_params = {
    "total_time": 2000,#200,
    "delta_t": 1.0,#0.05,
    "compression_factor": 10,
    "save_data": True, 
    "alpha_change" : 1.0,
    "harsh_data": False,
    "averaging_method": "Arithmetic",
    "phi_lower": 0.001,
    "phi_upper": 0.005,
    "N": 20,
    "M": 5,
    "K": 10,
    "prob_rewire": 0.2,#0.05,
    "set_seed": 1,
    "culture_momentum_real": 100,#5,
    "learning_error_scale": 0.02,
    "discount_factor": 0.8,
    "present_discount_factor": 0.99,
    "inverse_homophily": 0.2,#0.1,#1 is total mixing, 0 is no mixing
    "homophilly_rate" : 1,
    "confirmation_bias": -100,
    "alpha_attitude": 0.1,
    "beta_attitude": 0.1,
    "alpha_threshold": 1,
    "beta_threshold": 1,
}
base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

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
    {"property":"learning_error_scale","min":0.0,"max":0.5 , "title": r"$\epsilon$" },
    {"property":"alpha_attitude","min":0.1, "max": 8, "title": r"Attitude $\alpha$"}, 
    {"property":"beta_attitude","min":0.1, "max":8 , "title": r"Attitude $\beta$"}, 
    {"property":"alpha_threshold","min":0.1, "max": 8, "title": r"Threshold $\alpha$"}, 
    {"property":"beta_threshold","min":0.1, "max": 8, "title": r"Threshold $\beta$"}, 
    {"property":"discount_factor","min":0.0, "max":1.0 , "title": r"$\delta$"}, 
    {"property":"inverse_homophily","min":0.0, "max": 1.0, "title": r"$h$"}, 
    {"property":"present_discount_factor","min":0.0, "max": 1.0, "title": r"$\beta$"}, 
    {"property":"confirmation_bias","min":0.0, "max":100 , "title": r"$\theta$"}, 
]

#SOBOL
calc_second_order = False
print("calc_second_order: ", calc_second_order)

# Visualize
dpi_save = 1200

#modules
def sa_run(N_samples,base_params, variable_parameters_dict,calc_second_order):
    """
    Generate the list of dictionaries of parameters for different simulation runs and save the data from the sensitivity analysis

    Parameters
    ----------
    

    Returns
    -------
    
    """
    ##AVERAGE RUNS
    AV_reps = len(base_params["seed_list"])
    print("Average reps: ",AV_reps)
    problem, fileName, param_values = generate_problem(variable_parameters_dict,N_samples,AV_reps,calc_second_order)
    sa_save_problem(problem,fileName)

    params_list_sa = produce_param_list_SA(param_values,base_params,variable_parameters_dict)

    Y_emissions, Y_mu, Y_var,  Y_coefficient_of_variance = parallel_run_sa(params_list_sa)
    
    sa_save_Y(Y_emissions,fileName,"Y_emissions")
    sa_save_Y(Y_mu,fileName, "Y_mu")
    sa_save_Y(Y_var,fileName, "Y_var")
    sa_save_Y(Y_coefficient_of_variance,fileName, "Y_coefficient_of_variance")

    return AV_reps,problem, fileName, param_values, Y_emissions, Y_mu, Y_coefficient_of_variance

def generate_problem(variable_parameters_dict,N_samples,AV_reps,calc_second_order):
    """


    Parameters
    ----------
    

    Returns
    -------

    """
    D_vars = len(variable_parameters_dict)
    calc_second_order = False

    if calc_second_order:
        samples = N_samples * (2*D_vars + 2)
    else:
        samples = N_samples * (D_vars + 2)

    print("samples: ",samples)

    names_list = [x["property"] for x in variable_parameters_dict]
    bounds_list = [[x["min"],x["max"]] for x in variable_parameters_dict]

    problem = {
        "num_vars": D_vars,
        "names": names_list,
        "bounds": bounds_list,
    }

    ########################################

    fileName = "results/SA_%s_%s_%s_%s" % (str(AV_reps),str(samples),str(D_vars),str(N_samples))
    print("fileName: ", fileName)
    createFolderSA(fileName)
    
    # GENERATE PARAMETER VALUES
    param_values = saltelli.sample(
        problem, N_samples, calc_second_order = calc_second_order
    )  # NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters

    return problem, fileName, param_values

def produce_param_list_SA(param_values,params,variable_parameters_dict):
    "param_values are the satelli samples, params are the fixed variables, variable parameters is the list of SA variables, we want the name!"

    params_list = []
    for i, X in enumerate(param_values):
        variable_params_added = add_varaiables_to_dict(params,variable_parameters_dict,X)
        params_list.append(variable_params_added.copy())
    return params_list

def get_plot_data(problem, Y_emissions, Y_mu, Y_coefficient_of_variance,calc_second_order):
    """


    Parameters
    ----------
    

    Returns
    -------
    
    """    
    
    Si_emissions = sobol.analyze(problem, Y_emissions, calc_second_order = calc_second_order, print_to_console=False, )
    Si_mu = sobol.analyze(problem, Y_mu, calc_second_order = calc_second_order, print_to_console=False)
    Si_coefficient_of_variance = sobol.analyze(problem, Y_coefficient_of_variance, calc_second_order = calc_second_order, print_to_console=False)
    
    names = [x["title"] for x in variable_parameters_dict]

    #### Bar chart
    if calc_second_order: 
        total_emissions, first_emissions, second_emissions = Si_emissions.to_df()
        total_mu, first_mu, second_mu = Si_mu.to_df()
        total_coefficient_of_variance, first_coefficient_of_variance, second_coefficient_of_variance = Si_coefficient_of_variance.to_df()
    else:
        total_emissions, first_emissions = Si_emissions.to_df()
        total_mu, first_mu = Si_mu.to_df()
        total_coefficient_of_variance, first_coefficient_of_variance = Si_coefficient_of_variance.to_df()


    total_data_sa_emissions, total_yerr_emissions = get_data_bar_chart(total_emissions)
    total_data_sa_mu, total_yerr_mu = get_data_bar_chart(total_mu)
    total_data_sa_coefficient_of_variance, total_yerr_coefficient_of_variance = get_data_bar_chart(total_coefficient_of_variance)

    first_data_sa_emissions, first_yerr_emissions = get_data_bar_chart(first_emissions)
    first_data_sa_mu, first_yerr_mu = get_data_bar_chart(first_mu)
    first_data_sa_coefficient_of_variance, first_yerr_coefficient_of_variance = get_data_bar_chart(first_coefficient_of_variance)

    data_sa_dict_total = {
        "emissions" : {"data": total_data_sa_emissions, "yerr": total_yerr_emissions,},
        "mu" : { "data": total_data_sa_mu, "yerr": total_yerr_mu,},
        "coefficient_of_variance" : {"data": total_data_sa_coefficient_of_variance, "yerr": total_yerr_coefficient_of_variance},
    }
    data_sa_dict_first = {
        "emissions" : {"data": first_data_sa_emissions, "yerr": first_yerr_emissions,},
        "mu" : { "data": first_data_sa_mu, "yerr": first_yerr_mu,},
        "coefficient_of_variance" : {"data": first_data_sa_coefficient_of_variance, "yerr": first_yerr_coefficient_of_variance},
    }

    return names, data_sa_dict_total, data_sa_dict_first

def get_data_bar_chart(Si_df):
    """


    Parameters
    ----------
    Si_df: pd.DataFrame, 
        Dataframe of sensitivity results. Taken from: https://salib.readthedocs.io/en/latest/_modules/SALib/plotting/bar.html

    Returns
    -------
    
    """

    # magic string indicating DF columns holding conf bound values
    conf_cols = Si_df.columns.str.contains('_conf')
    confs = Si_df.loc[:, conf_cols]#select all those that ARE in conf_cols!
    confs.columns = [c.replace('_conf', '') for c in confs.columns]
    Sis = Si_df.loc[:, ~conf_cols]#select all those that ARENT in conf_cols!

    return Sis,confs

def Merge_dict_SA(data_sa_dict, plot_dict):
    """


    Parameters
    ----------
    

    Returns
    -------
    
    """    
    for i in data_sa_dict.keys():
        for v in plot_dict[i].keys():
            data_sa_dict[i][v]=plot_dict[i][v]
    return data_sa_dict

if __name__ == "__main__":
    RUN = 0#False,True
    if RUN:
        N_samples = 512
        AV_reps,problem, fileName, param_values, Y_emissions, Y_mu, Y_coefficient_of_variance = sa_run(N_samples,base_params, variable_parameters_dict,calc_second_order)
    else:
        fileName = "results/SA_5_8192_14_512"
        N_samples = 4
        problem = sa_load_problem(fileName)
        Y_emissions = sa_load_Y(fileName, "Y_emissions")
        Y_mu = sa_load_Y(fileName, "Y_mu")
        Y_coefficient_of_variance = sa_load_Y(fileName, "Y_coefficient_of_variance")

    names, data_sa_dict_total, data_sa_dict_first  = get_plot_data(problem, Y_emissions, Y_mu, Y_coefficient_of_variance,calc_second_order)

    plot_dict = {
        "emissions" : {"title": r"$E/NM$",  "colour": "r", "linestyle": "--"},
        "mu" : {"title": r"$\mu/NM$", "colour": "g", "linestyle": "-"},
        "coefficient_of_variance" : {"title": r"$\sigma NM/\mu$", "colour": "b","linestyle": "-."},
    }

    #print("test", data_sa_dict_total)
    data_sa_dict_total = Merge_dict_SA(data_sa_dict_total, plot_dict)
    data_sa_dict_first = Merge_dict_SA(data_sa_dict_first, plot_dict)

    #multi_scatter_total_sensitivity_analysis_plot(fileName, data_sa_dict_total, names, dpi_save,N_samples, "Total")
    #multi_scatter_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, names, dpi_save,N_samples, "First")

    #multi_scatter_sidebyside_total_sensitivity_analysis_plot(fileName, data_sa_dict_total, names, dpi_save,N_samples, "Total")
    #multi_scatter_sidebyside_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, names, dpi_save,N_samples, "First")
    multi_scatter_separate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, names, dpi_save,N_samples, "First")

    plt.show()