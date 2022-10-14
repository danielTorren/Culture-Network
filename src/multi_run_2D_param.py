"""Run multiple simulations varying two parameters
A module that use input data to generate data from multiple social networks varying two properties
between simulations so that the differences may be compared. These multiple runs can either be single 
shot runs or taking the average over multiple runs. Useful for comparing the influences of these parameters
on each other to generate phase diagrams.

TWO MODES 
    The two parameters can be varied covering a 2D plane of points. This can either be done in SINGLE = True where individual 
    runs are used as the output and gives much greater variety of possible plots but all these plots use the same initial
    seed. Alternatively can be run such that multiple averages of the simulation are produced and then the data accessible 
    is the emissions, mean identity, variance of identity and the coefficient of variance of identity.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

#imports
from logging import raiseExceptions
from run import parallel_run, parallel_run_sa
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import json
from matplotlib.colors import LinearSegmentedColormap,  Normalize
from matplotlib.cm import get_cmap
from network import Network
from plot import (
    live_print_culture_timeseries,
    print_culture_timeseries_vary_array,
    live_print_culture_timeseries_vary,
    live_compare_plot_animate_behaviour_scatter,
    live_compare_animate_culture_network_and_weighting,
    live_compare_animate_weighting_matrix,
    live_compare_animate_behaviour_matrix,
    #live_print_heterogenous_culture_momentum_double,
    live_average_multirun_double_phase_diagram_mean,
    live_average_multirun_double_phase_diagram_mean_alt,
    live_average_multirun_double_phase_diagram_C_of_V_alt,
    live_average_multirun_double_phase_diagram_C_of_V,
)
from utility import (
    createFolder,
    generate_vals_variable_parameters_and_norms,
    save_object,
    load_object,
)

#run bools
RUN = 1#run or load in previously saved data
SINGLE = 0#determine if you runs single shots or study the averages over multiple runs for each experiment

###PLOT STUFF
dpi_save = 1200
round_dec = 2
cmap_weighting = "Reds"
cmap_edge = get_cmap("Greys")
fps = 5
interval = 50
layout = "circular"
node_size = 50
cmap = LinearSegmentedColormap.from_list("BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1)
norm_zero_one  = Normalize(vmin=0, vmax=1)

#modules
def produce_param_list_n_double(params_dict: dict,variable_parameters_dict: dict[dict]) -> list[dict]:
    """Creates a list of the param dictionaries. This only varies both parameters at the same time in a grid like fashion.
    
    Parameters
    ----------
    params_dict: dict,
        dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters. 
        e.g
            params = {
                "set_seed": 1,
                "total_time": 150,#200,
                "delta_t": 1.0,#0.05,
                "compression_factor": 10,
                "save_data": True, 
                "alpha_change" : 1.0,
                "harsh_data": False,
                "averaging_method": "Arithmetic",
                "phi_lower": 0.001,
                "phi_upper": 0.005,
                "N": 50,
                "M": 5,
                "K": 10,
                "prob_rewire": 0.2,#0.05,
                "culture_momentum_real": 100,#5,
                "learning_error_scale": 0.02,
                "discount_factor": 0.99,
                "present_discount_factor": 0.95,
                "inverse_homophily": 0.2,#0.1,#1 is total mixing, 0 is no mixing
                "homophilly_rate" : 1,
                "confirmation_bias": 50,
                "alpha_attitude": 0.5,
                "beta_attitude": 0.5,
                "alpha_threshold": 1,
                "beta_threshold": 1,
            }
            params["time_steps_max"] = int(params["total_time"] / params["delta_t"])
    variable_parameters_dict: dict[dict]
        dictionary of dictionaries containing details for range of parameters to vary. e.g.
            variable_parameters_dict = {
                "inverse_homophily": {"position": "row","property":"inverse_homophily","min":0.0, "max": 1.0, "title": r"$h$", "divisions": "log", "reps": 2}, 
                "confirmation_bias": {"position": "col","property":"confirmation_bias","min":0, "max":30 , "title": r"$\theta$", "divisions": "linear", "reps": 2}, 
            }
    
    Returns
    -------
    params_list: list[dict]
        list of parameter dicts, each entry corresponds to one experiment to be tested
    """

    params_list = []

    for i in variable_parameters_dict["row"]["vals"]:
        for j in variable_parameters_dict["row"]["vals"]:
            params_dict[variable_parameters_dict["row"]["property"]] = i
            params_dict[variable_parameters_dict["col"]["property"]] = j
            params_list.append(params_dict.copy()) 
    
    return params_list

def generate_title_list(
    title_col: str,
    col_list: list,
    title_row: str,
    row_list: list,
    round_dec: float,
    ) -> list[str]:
    """Generate a list of title to be used in the plots
    
    Parameters
    ----------
    title_col: str
        title of property varied in the col direction
    col_list: list
        list of values for property varied in the col direction
    title_row: str
        title of property varied in the row direction
    row_list: list
        list of values for property varied in the row direction
    round_dec:
        number of decimals to round float values too
    
    Returns
    -------
    title_list: list[str]
        list of titles one for each experiment
    """
    title_list = []
    
    for i in range(len(row_list)):
        for j in range(len(col_list)):
            title_list.append(("%s = %s, %s = %s") % (title_row,str(round(row_list[i],round_dec)), title_col,str(round(col_list[j], round_dec))))
    
    return  title_list

def shot_two_dimensional_param_run(fileName: str,params: dict,variable_parameters_dict: dict[dict],reps_row: int,reps_col: int) -> tuple[list[Network], npt.NDArray,list[str]]:
    """Generate results for the case of varying two parameters in a single shot for each experiment, also create folder 
    and titles for plots and save data
    
    Parameters
    ----------
    fileName: str
        where to save it e.g in the results folder in data or plots folder
    params: dict
        dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters.
    variable_parameters_dict: dict[dict]
        dictionary of dictionaries containing details for range of parameters to vary. see produce_param_list_n_double for an example
    reps_row: int
        repetitions along the row direction of experiment e.g. if i want to vary from 0 - 10, how may divisions to make
    reps_col: int
        repetitions along the col direction of experiment
    Returns
    -------
    data_list: list[Network]
        list of network results
    data_array: npt.NDArray
        array of network results, shape row x cols
    title_list: list[str]
        list of titles one for each experiment    
    """
    createFolder(fileName)    
    params_dict_list = produce_param_list_n_double(params,variable_parameters_dict)
    
    data_list = parallel_run(params_dict_list) 
    data_array = np.reshape(data_list, (reps_row, reps_col))

    save_data_shot(fileName, variable_parameters_dict, data_list, data_array)

    return data_list, data_array

def av_two_dimensional_param_run(fileName: str, variable_parameters_dict: dict[dict], params: dict) -> tuple[ npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Generate results for the case of varying two parameters in multiple stochastically averaged runs for each experiment, also create folder 
    and titles for plots and save data
    
    Parameters
    ----------
    fileName: str
        where to save it e.g in the results folder in data or plots folder
    variable_parameters_dict: dict[dict]
        dictionary of dictionaries containing details for range of parameters to vary. See produce_param_list_n_double for an example
    params: dict
        dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters. See produce_param_list_n_double for an example
    
    Returns
    -------
    results_emissions: npt.NDArray
        Array of normalized societal emissions at the end of the allowed time, normalize by number of agents and behaviours per agent. 
    results_mean: npt.NDArray
        Array of normalized mean societal identity
    results_var: npt.NDArray
        Array of variance of societal identity for different parameter runs 
    results_coefficient_variance: npt.NDArray
        Array of coefficient of variance (std/mu) of societal identity for different parameter runs
    """

    createFolder(fileName)
    params_list = produce_param_list_n_double(params,variable_parameters_dict)
    results_emissions, results_mu, results_var, results_coefficient_of_variance = parallel_run_sa(params_list)
    
    #save the data and params_list
    save_data_av(fileName, results_emissions, results_mu, results_var, results_coefficient_of_variance)

    return results_emissions, results_mu, results_var, results_coefficient_of_variance

def reshape_results_matricies(results_emissions, results_mu, results_var, results_coefficient_of_variance, reps_row, reps_col):
    """Reshape lists of results into matricies for the save of plotting in Array form"""

    matrix_emissions = results_emissions.reshape((reps_row, reps_col))
    matrix_mu = results_mu.reshape((reps_row, reps_col))
    matrix_var = results_var.reshape((reps_row, reps_col))
    matrix_coefficient_of_variance = results_coefficient_of_variance.reshape((reps_row, reps_col))

    return matrix_emissions, matrix_mu, matrix_var, matrix_coefficient_of_variance

def get_params(variable_parameters_dict):
    """unpacking data from the variable_parameters_dict to be used in the generation of results, makes accessing the property easier"""
    for i in variable_parameters_dict.values():
        if i["position"] == "row":
            param_row = i["property"]
        elif i["position"] == "col":
            param_col = i["property"]
        else:
            raiseExceptions("Invalid position in variable_parameters_dict entry")

    reps_row = variable_parameters_dict[param_row]["reps"]
    reps_col = variable_parameters_dict[param_col]["reps"]
    reps = reps_row*reps_col

    property_row = variable_parameters_dict[param_row]["title"]
    property_col = variable_parameters_dict[param_col]["title"]
    
    param_min_row = variable_parameters_dict[param_row]["min"]
    param_max_row = variable_parameters_dict[param_row]["max"]
    param_min_col = variable_parameters_dict[param_col]["min"]
    param_max_col = variable_parameters_dict[param_col]["max"]

    property_varied_values_row = variable_parameters_dict[param_row]["vals"]
    property_varied_values_col = variable_parameters_dict[param_col]["vals"]

    return param_row, param_col, reps_row, reps_col, reps, property_row, property_col, param_min_row, param_max_row , param_min_col, param_max_col, property_varied_values_row, property_varied_values_col

def save_data_shot(fileName, variable_parameters_dict, data_list, data_array):
    """save variable_parameters_dict and data from single shot experiments"""
    save_object(variable_parameters_dict,fileName + "/Data", "variable_parameters_dict")
    save_object(data_list,fileName + "/Data","data_list")
    save_object(data_array,fileName + "/Data","data_array")

def save_data_av(fileName, results_emissions, results_mu, results_var, results_coefficient_of_variance):
    """save variable_parameters_dict and results from the stochastically averaged runs"""
    save_object(variable_parameters_dict,fileName + "/Data", "variable_parameters_dict")
    save_object(results_emissions,fileName + "/Data","results_emissions")
    save_object(results_mu,fileName + "/Data", "results_mu")
    save_object(results_var,fileName + "/Data", "results_var")
    save_object(results_coefficient_of_variance,fileName + "/Data", "results_coefficient_of_variance")

def load_data_shot(fileName):
    """load variable_parameters_dict and data from single shot experiments"""
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    data_list = load_object(fileName + "/Data","data_list")
    data_array = load_object(fileName + "/Data","data_array")

    return variable_parameters_dict, data_list, data_array

def load_data_av(fileName):
    """load variable_parameters_dict and results from the stochastically averaged runs"""
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    results_emissions = load_object(fileName + "/Data",  "results_emissions")
    results_mu = load_object(fileName + "/Data",  "results_mu")
    results_var = load_object(fileName + "/Data",  "results_var")
    results_coefficient_of_variance = load_object(fileName + "/Data",  "results_coefficient_of_variance")

    return variable_parameters_dict, results_emissions, results_mu, results_var, results_coefficient_of_variance

if __name__ == "__main__":
    if SINGLE:
        if RUN:
            #load base params
            f_base_params = open("src/constants/base_params.json")
            params = json.load(f_base_params)
            f_base_params.close()
            params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

            #load variable params
            f_variable_parameters = open("src/constants/variable_parameters_dict_2D.json")
            variable_parameters_dict = json.load(f_variable_parameters)
            f_variable_parameters.close()

            variable_parameters_dict = generate_vals_variable_parameters_and_norms(variable_parameters_dict)

            fileName = "results/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (variable_parameters_dict["col"]["property"],variable_parameters_dict["row"]["property"],str(params["N"]),str(params["time_steps_max"]),str(params["K"]),str(variable_parameters_dict["col"]["min"]), str(variable_parameters_dict["col"]["max"]), str(variable_parameters_dict["row"]["min"]), str(variable_parameters_dict["row"]["max"]), variable_parameters_dict["col"]["reps"],variable_parameters_dict["row"]["reps"])
            print("fileName: ", fileName)

            title_list = generate_title_list(variable_parameters_dict["col"]["title"],variable_parameters_dict["col"]["vals"],variable_parameters_dict["row"]["title"],variable_parameters_dict["row"]["vals"], round_dec)
            data_list, data_array = shot_two_dimensional_param_run(fileName,params,variable_parameters_dict,variable_parameters_dict["row"]["reps"],variable_parameters_dict["col"]["reps"])

        else:
            fileName = "results/beta_attitude_alpha_attitude_50_2000_10_-1_1_-1_1_3_3"
            variable_parameters_dict, data_list, data_array = load_data_shot(fileName)
            
            title_list = generate_title_list(variable_parameters_dict["col"]["title"],variable_parameters_dict["col"]["vals"],variable_parameters_dict["row"]["title"],variable_parameters_dict["row"]["vals"], round_dec)

        ### PLOTS FOR SINGLE SHOT RUNS
        live_print_culture_timeseries_vary(fileName, data_list, variable_parameters_dict["row"]["property"], variable_parameters_dict["col"]["property"],title_list, variable_parameters_dict["row"]["reps"], variable_parameters_dict["col"]["reps"],  dpi_save)
        #BROKEN print_culture_timeseries_vary_array(fileName, data_array, param_col,property_col,property_varied_values_col,param_row, property_row,property_varied_values_row,  reps_row, reps_col , dpi_save)

        #ani_b = live_compare_animate_culture_network_and_weighting(fileName,data_list,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec,cmap_edge, reps_col, reps_row,property_col,property_varied_values_col)
        #ani_c = live_compare_animate_weighting_matrix(fileName, data_list,  cmap_weighting, interval, fps, round_dec, cmap_edge, reps_row, reps_col,property_col,property_varied_values_col)
        #ani_d = live_compare_animate_behaviour_matrix(fileName, data_list,  cmap, interval, fps, round_dec, reps_row, reps_col,property_col,property_varied_values_col)

    else: 
        if RUN:
            #load base params
            f_base_params = open("src/constants/base_params.json")
            params = json.load(f_base_params)
            f_base_params.close()
            params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

            #load variable params
            f_variable_parameters = open("src/constants/variable_parameters_dict_2D.json")
            variable_parameters_dict = json.load(f_variable_parameters)
            f_variable_parameters.close()

            #AVERAGE OVER MULTIPLE RUNS           
            variable_parameters_dict = generate_vals_variable_parameters_and_norms(variable_parameters_dict)

            fileName = "results/average_%s_%s_%s_%s_%s_%s_%s_%s" % (variable_parameters_dict["col"]["property"],variable_parameters_dict["row"]["property"],str(params["N"]),str(params["time_steps_max"]),str(params["K"]),str(variable_parameters_dict["col"]["reps"]),str(variable_parameters_dict["row"]["reps"]), len(params["seed_list"]))
            print("fileName: ", fileName)

            results_emissions, results_mu, results_var, results_coefficient_of_variance = av_two_dimensional_param_run(fileName, variable_parameters_dict, params)
        else:
            fileName = "results/beta_attitude_alpha_attitude_50_2000_10_-1_1_-1_1_3_3"
            createFolder(fileName)
            
            variable_parameters_dict,results_emissions, results_mu, results_var, results_coefficient_of_variance = load_data_av(fileName)

        ###PLOTS FOR STOCHASTICALLY AVERAGED RUNS
        matrix_emissions, matrix_mu, matrix_var, matrix_coefficient_of_variance = reshape_results_matricies(results_emissions, results_mu, results_var, results_coefficient_of_variance,variable_parameters_dict["row"]["reps"], variable_parameters_dict["col"]["reps"])

        live_average_multirun_double_phase_diagram_mean_alt(fileName, matrix_mu, variable_parameters_dict, get_cmap("Blues"),dpi_save)
        live_average_multirun_double_phase_diagram_C_of_V_alt(fileName, matrix_mu, variable_parameters_dict, get_cmap("Reds"),dpi_save)
        #live_average_multirun_double_phase_diagram_mean(fileName, matrix_mu, property_row, property_varied_values_row,property_col,property_varied_values_col, get_cmap("Blues"),dpi_save,round_dec)
        #live_average_multirun_double_phase_diagram_C_of_V(fileName, matrix_coefficient_of_variance, variable_parameters_dict["row"]["property"], variable_parameters_dict["row"]["vals"],variable_parameters_dict["col"]["property"],variable_parameters_dict["col"]["vals"], get_cmap("Reds"),dpi_save,round_dec)

    plt.show()