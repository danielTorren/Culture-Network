"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. These multiple runs can either be single 
shot runs or taking the average over multiple runs.

TWO MODES 
    Single parameters can be varied to cover a list of points. This can either be done in SINGLE = True where individual 
    runs are used as the output and gives much greater variety of possible plots but all these plots use the same initial
    seed. Alternatively can be run such that multiple averages of the simulation are produced and then the data accessible 
    is the emissions, mean identity, variance of identity and the coefficient of variance of identity.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import json
import numpy as np
from resources.utility import createFolder,produce_name_datetime,save_object
from resources.run import parallel_run, parallel_run_sa,culture_data_run

# modules
def produce_param_list(params: dict, property_list: list, property: str) -> list[dict]:
    """
    Produce a list of the dicts for each experiment

    Parameters
    ----------
    params: dict
        base parameters from which we vary e.g
        params = {
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
            params["time_steps_max"] = int(params["total_time"] / params["delta_t"])
    porperty_list: list
        list of values for the property to be varied
    property: str
        property to be varied

    Returns
    -------
    params_list: list[dict]
        list of parameter dicts, each entry corresponds to one experiment to be tested
    """

    params_list = []
    for i in property_list:
        params[property] = i
        params_list.append(
            params.copy()
        )  # have to make a copy so that it actually appends a new dict and not just the location of the params dict
    return params_list

def main(RUN_TYPE = 0) -> str: 
    nrows_gen = 2 
    ncols_gen = 3
    """The number of rows and cols set the number of experiments ie 4 rows and 3 cols gives 12 experiments"""
    reps = nrows_gen * ncols_gen 

    if RUN_TYPE == 0:
        #FOR POLARISATION A,B PLOT
        property_varied = "a_attitude"
        property_varied_title = "Attitude Beta parameters, (a,b)"
        param_min = 0.05
        param_max = 2.0  # 50.0
        property_values_list = np.asarray([0.05, 0.3, 2.0])
        title_list = [r"Attitude Beta parameters, (a,b) = 0.05, Confirmation bias, $\theta$ = 40", r"Attitude Beta parameters, (a,b) = 0.5, Confirmation bias, $\theta$ = 20", r"Attitude Beta parameters, (a,b) = 2, Confirmation bias, $\theta$ = 10"]
    elif RUN_TYPE == 1:
        ###############################
        #FOR HOMOPHILY PLOT
        property_varied = "homophily"
        property_varied_title = "Attribute homophily, $h$"
        param_min = 0.0
        param_max = 1.0  # 50.0
        title_list = [r"Homophily, h = 0.0", r"Homophily, h = 0.5", r"Homophily, h = 1.0"]
        property_values_list = np.asarray([0.1, 0.5, 1.0])
        property_varied = "confirmation_bias"
        property_varied_title = "Confirmation bias $\theta$"
        param_min = -0.0
        param_max = 100.0  # 50.0
        reps = 500
        title_list = ["Bifurcation"]
        #title_list = [r"Confirmation bias $\theta$ = 1.0", r"Confirmation bias $\theta$ = 10.0", r"Confirmation bias $\theta$ = 25.0", r"Confirmation bias $\theta$ = 50.0"]
        property_values_list = np.linspace(param_min,param_max, reps)
        print("property_values_list ", property_values_list )

    f = open("constants/base_params.json")
    params = json.load(f)
    params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

    root = "one_param_sweep_single"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    if RUN_TYPE == 0:
        case_1 = params.copy()
        case_2 = params.copy()
        case_3 = params.copy()

        case_3["a_attitude"] = 0.05
        case_3["b_attitude"] = 0.05
        case_3["confirmation_bias"] = 40

        case_2["a_attitude"] = 0.3
        case_2["b_attitude"] = 0.3  
        case_2["confirmation_bias"] = 18#THIS IS NEEDS TO SPLIT PARALLEL

        case_1["a_attitude"] = 2.0
        case_1["b_attitude"] = 2.0
        case_1["confirmation_bias"] = 10

        params_list = [case_1, case_2, case_3]
        print(params_list)
    else:
        params_list = produce_param_list(params, property_values_list, property_varied)

    data_list = parallel_run(params_list)  # better if a Multiple of 4
    createFolder(fileName)

    save_object(data_list, fileName + "/Data", "data_list")
    save_object(params, fileName + "/Data", "base_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(param_min, fileName + "/Data", "param_min")
    save_object(param_max, fileName + "/Data", "param_max")
    save_object(title_list, fileName + "/Data", "title_list")
    save_object(property_values_list, fileName + "/Data", "property_values_list")
    if RUN_TYPE == 0:
        #FOR POLARISATION A,B PLOT - NEED TO SET self.b_attitude = parameters["a_attitude"] in NETWORK
        property_varied = "a_attitude"
        property_varied_title = "Attitude Beta parameters, (a,b)"
        param_min = 0.05
        param_max = 2.0  # 50.0
        property_values_list = np.asarray([0.05, 0.5, 2.0])# FOR ALPHA
        title_list = [r"Attitude Beta parameters, (a,b) = 0.05, Confirmation bias, $\theta$ = 40", r"Attitude Beta parameters, (a,b) = 0.5, Confirmation bias, $\theta$ = 20", r"Attitude Beta parameters, (a,b) = 2, Confirmation bias, $\theta$ = 10"]
    elif RUN_TYPE == 1:
        ################
        #FOR ALPHA CHANGE PLOT
        property_varied = "alpha_change"
        property_varied_title = "alpha_change"
        param_min = 0.0
        param_max = 1.0  # 50.0
        title_list = [r"Static uniform $\alpha_{n,k}$", r"Static culturally determined $\alpha_{n,k}$", r"Dynamic culturally determined $\alpha_{n,k}$"]
        property_values_list = np.asarray([0.0, 0.5, 1.0])# FOR ALPHA
    elif RUN_TYPE == 2:
        ###############################
        #FOR HOMOPHILY PLOT
        property_varied = "homophily"
        property_varied_title = "Attribute homophily, $h$"
        param_min = 0.0
        param_max = 1.0  # 50.0
        title_list = [r"Homophily, h = 0.0", r"Homophily, h = 0.5", r"Homophily, h = 1.0"]
        property_values_list = np.asarray([0.2, 0.6, 1.0])
    elif RUN_TYPE == 3:
        ###############################
        #FOR NETOWRK STRUCTURE HOMOPHILY PLOT
        property_varied = "homophily"
        property_varied_title = "Attribute homophily, $h$"
        param_min = 0.0
        param_max = 1.0  # 50.0
        title_list = [r"Small world, Homophily, h = 0.0", r"Small world, Homophily, h = 0.5", r"Small world, Homophily, h = 1.0",r"Scale free, Homophily, h = 0.0", r"Scale free, Homophily, h = 0.5", r"Scale free, Homophily, h = 1.0"]
        property_values_list = np.asarray([0.2, 0.6, 1.0, 0.2, 0.6, 1.0])
    elif RUN_TYPE == 4:
        property_varied = "guilty_individual_power"
        property_varied_title = "Identity power"
        param_min = 1.0
        param_max = 20.0  # 50.0
        title_list = [r"Identity power = 1.0", r"Identity power = 5.0", r"Identity power = 10.0", r"Identity power = 20.0"]
        property_values_list = np.asarray([1.0, 5.0, 10.0, 20.0])
    elif RUN_TYPE == 5:
        property_varied = "confirmation_bias"
        property_varied_title = "Confirmation bias $\theta$"
        param_min = 0.0
        param_max = 50.0  # 50.0
        title_list = [r"Confirmation bias $\theta$ = 1.0", r"Confirmation bias $\theta$ = 10.0", r"Confirmation bias $\theta$ = 25.0", r"Confirmation bias $\theta$ = 50.0"]
        property_values_list = np.asarray([0.0, 10.0, 25.0, 50.0])
    elif RUN_TYPE == 6:
        ######### REMEMBER TO SET SAVE TO 0!!
        property_varied = "green_N"
        property_varied_title = "Number of eco-warriors"
        param_min = 0.0
        param_max = 64.0  # 50.0
        title_list = ["Impact of eco-warriors on final identity distribution"]
        property_values_list =  np.asarray([0, 2, 4, 8, 12, 16, 32, 64])#np.arange(8)
    elif RUN_TYPE == 7:
        ######### REMEMBER TO SET SAVE TO 0!!
        property_varied = "confirmation_bias"
        property_varied_title = "Confirmation bias $\theta$"
        param_min = 0.0
        param_max = 50.0  # 50.0
        title_list = ["Impact of eco-warriors on final identity distribution"] 
        property_values_list = np.asarray([0, 2, 4, 8, 12, 16, 32, 64])

    f = open("constants/base_params.json")
    params = json.load(f)
    params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

    root = "one_param_sweep_multi"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    dataName = fileName + "/Data"

    params_list = produce_param_list(params, property_values_list, property_varied)

    if RUN_TYPE == 6 or 7:
        (
            results_culture_lists
            
        ) = culture_data_run(params_list)
        
        createFolder(fileName)

        save_object(results_culture_lists, fileName + "/Data", "results_culture_lists")
    else:
        (
            results_emissions,
            results_mu,
            results_var,
            results_coefficient_of_variance,
            results_emissions_change,
        ) = parallel_run_sa(params_list)
        
        createFolder(fileName)

        save_object(results_emissions, fileName + "/Data", "results_emissions")
        save_object(results_mu, fileName + "/Data", "results_mu")
        save_object(results_var, fileName + "/Data", "results_var")
        save_object( results_coefficient_of_variance, fileName + "/Data","results_coefficient_of_variance")
        save_object(results_emissions_change, fileName + "/Data", "results_emissions_change")

    save_object(params, fileName + "/Data", "base_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(param_min, fileName + "/Data", "param_min")
    save_object(param_max, fileName + "/Data", "param_max")
    save_object(title_list, fileName + "/Data", "title_list")
    save_object(property_values_list, fileName + "/Data", "property_values_list")

    return fileName
