from utility import (
    sa_save_problem,
    sa_save_Y,
    sa_load_problem,
    sa_load_Y,
    createFolderSA,
    produce_param_list_SA
)
from run import parallel_run_sa, average_seed_parallel_run_sa,average_seed_parallel_run_sa_emmissions_mu_variance
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import random
from datetime import datetime
import matplotlib.pyplot as plt
from plot import (
    prints_SA_matrix,
    bar_sensitivity_analysis_plot,
    scatter_total_sensitivity_analysis_plot,
    multi_scatter_total_sensitivity_analysis_plot,
    multi_scatter_sidebyside_total_sensitivity_analysis_plot,
    multi_scatter_seperate_total_sensitivity_analysis_plot
)
#print("Hey",random.seed(datetime.now()))

base_params = {
    "seed_list": [1,2,3,4,5],
    "total_time": 30,
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
    "K": 15,
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
    base_params["alpha_attitude"] = 1
    base_params["beta_attitude"] = 1
    base_params["alpha_threshold"] = 1
    base_params["beta_threshold"] = 1

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

def average_generate_problem(variable_parameters_dict,N_samples,average_reps,calc_second_order):

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

    fileName = "results/average_SA_%s_%s_%s_%s" % (str(average_reps),str(samples),str(D_vars),str(N_samples))
    print("fileName: ", fileName)
    createFolderSA(fileName)
    
    # GENERATE PARAMETER VALUES
    param_values = saltelli.sample(
        problem, N_samples, calc_second_order = calc_second_order
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

def average_generate_sa_data_emmissions_mu_variance(base_params,variable_parameters_dict,param_values):

    params_list_sa = produce_param_list_SA(param_values,base_params,variable_parameters_dict)

    average_Y_emissions, average_Y_mu, average_Y_coefficient_of_variance = average_seed_parallel_run_sa_emmissions_mu_variance(params_list_sa)
    
    return average_Y_emissions, average_Y_mu, average_Y_coefficient_of_variance

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

def emmissions_mu_variance_run(N_samples,base_params, variable_parameters_dict,calc_second_order):
        ##AVERAGE RUNS
        average_reps = len(base_params["seed_list"])
        print("Average reps: ",average_reps)
        average_problem, average_fileName, average_param_values = average_generate_problem(variable_parameters_dict,N_samples,average_reps,calc_second_order)
        sa_save_problem(average_problem,average_fileName)
        average_Y_emissions, average_Y_mu, average_Y_coefficient_of_variance  = average_generate_sa_data_emmissions_mu_variance(base_params,variable_parameters_dict,average_param_values)
        
        sa_save_Y(average_Y_emissions,average_fileName,"Y_emissions")
        sa_save_Y(average_Y_mu,average_fileName, "Y_mu")
        sa_save_Y(average_Y_coefficient_of_variance,average_fileName, "Y_coefficient_of_variance")
        return average_reps,average_problem, average_fileName, average_param_values, average_Y_emissions, average_Y_mu, average_Y_coefficient_of_variance

def get_emissions_mu_variance_plot_data(average_problem, average_Y_emissions, average_Y_mu, average_Y_coefficient_of_variance,calc_second_order):
        #Si = sobol.analyze(problem, Y, print_to_console=False)
        Si_emissions = sobol.analyze(average_problem, average_Y_emissions, calc_second_order = calc_second_order, print_to_console=False, )
        Si_mu = sobol.analyze(average_problem, average_Y_mu, calc_second_order = calc_second_order, print_to_console=False)
        Si_coefficient_of_variance = sobol.analyze(average_problem, average_Y_coefficient_of_variance, calc_second_order = calc_second_order, print_to_console=False)

        ###PLOT RESULTS
        
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


def emmissions_run(N_samples,base_params, variable_parameters_dict,calc_second_order):
    results_property = "Carbon Emissions/NM"

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
    average_problem, average_fileName, average_param_values = average_generate_problem(variable_parameters_dict,N_samples,average_reps,calc_second_order)
    sa_save_problem(average_problem,average_fileName)
    average_Y = average_generate_sa_data(base_params,variable_parameters_dict,average_param_values,results_property)
    sa_save_Y(average_Y,average_fileName)

    return average_reps,average_problem, average_fileName, average_param_values, average_Y

def get_emissions_plot_data(average_problem,average_Y):
    Si = sobol.analyze(average_problem, average_Y, print_to_console=False)

    ###PLOT RESULTS
    names = [x["title"] for x in variable_parameters_dict]
    #### Bar chart
    total, first, second = Si.to_df()
    data_sa, yerr = get_data_bar_chart(total)
    return data_sa, names, yerr, Si

def Merge_dict_SA(data_sa_dict, plot_dict):
    for i in data_sa_dict.keys():
        for v in plot_dict[i].keys():
            data_sa_dict[i][v]=plot_dict[i][v]
    return data_sa_dict

if __name__ == "__main__":
    EMISSIONS_SA = 0 
    EMISSIONS_MU_COEFFCIENT_VARIANCE_SA = 1

    if EMISSIONS_SA: 
        RUN = 0#False,True
        if RUN:
            N_samples = 1024#256
            average_reps,average_problem, average_fileName, average_param_values, average_Y = emmissions_run(N_samples,base_params, variable_parameters_dict,calc_second_order)

        else:
            average_fileName = "results/average_SA_4_30720_14_1024"
            N_samples = 256
            average_problem = sa_load_problem(average_fileName)
            average_Y = sa_load_Y(average_fileName, "Y")
        
        data_sa, names, yerr, Si = get_emissions_plot_data(average_problem,average_Y)

        scatter_total_sensitivity_analysis_plot(average_fileName, data_sa, names, yerr , dpi_save,N_samples)

        #Matrix plot
        data = [np.asarray(Si["S2"]),np.asarray(Si["S2_conf"])]
        title_list = ["S2","S2_conf"]
        cmap = "Blues"
        nrows = 1
        ncols = 2
        #prints_SA_matrix(average_fileName, data,title_list,cmap,nrows, ncols, dpi_save, names,N_samples)
    elif EMISSIONS_MU_COEFFCIENT_VARIANCE_SA:
        RUN = 0#False,True
        if RUN:
            N_samples = 512
            average_reps,average_problem, average_fileName, average_param_values, average_Y_emissions, average_Y_mu, average_Y_coefficient_of_variance = emmissions_mu_variance_run(N_samples,base_params, variable_parameters_dict,calc_second_order)
        else:
            average_fileName = "results/average_SA_5_8192_14_512"
            N_samples = 4
            average_problem = sa_load_problem(average_fileName)
            average_Y_emissions = sa_load_Y(average_fileName, "Y_emissions")
            average_Y_mu = sa_load_Y(average_fileName, "Y_mu")
            average_Y_coefficient_of_variance = sa_load_Y(average_fileName, "Y_coefficient_of_variance")

        names, data_sa_dict_total, data_sa_dict_first  = get_emissions_mu_variance_plot_data(average_problem, average_Y_emissions, average_Y_mu, average_Y_coefficient_of_variance,calc_second_order)

        plot_dict = {
            "emissions" : {"title": r"$E/NM$",  "colour": "r", "linestyle": "--"},
            "mu" : {"title": r"$\mu/NM$", "colour": "g", "linestyle": "-"},
            "coefficient_of_variance" : {"title": r"$\sigma NM/\mu$", "colour": "b","linestyle": "-."},
        }

        #print("test", data_sa_dict_total)
        data_sa_dict_total = Merge_dict_SA(data_sa_dict_total, plot_dict)
        data_sa_dict_first = Merge_dict_SA(data_sa_dict_first, plot_dict)

        #multi_scatter_total_sensitivity_analysis_plot(average_fileName, data_sa_dict_total, names, dpi_save,N_samples, "Total")
        #multi_scatter_total_sensitivity_analysis_plot(average_fileName, data_sa_dict_first, names, dpi_save,N_samples, "First")

        #multi_scatter_sidebyside_total_sensitivity_analysis_plot(average_fileName, data_sa_dict_total, names, dpi_save,N_samples, "Total")
        #multi_scatter_sidebyside_total_sensitivity_analysis_plot(average_fileName, data_sa_dict_first, names, dpi_save,N_samples, "First")
        multi_scatter_seperate_total_sensitivity_analysis_plot(average_fileName, data_sa_dict_first, names, dpi_save,N_samples, "First")
    else:
        print("NO RUN!")


    plt.show()