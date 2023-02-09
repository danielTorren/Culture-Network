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
from resources.run import parallel_run

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
                "alpha_change" : "C",
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

    f = open("constants/base_params.json")
    params = json.load(f)

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

    return fileName
