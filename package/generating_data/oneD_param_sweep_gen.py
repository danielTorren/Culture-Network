"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. 




Created: 10/10/2022
"""

# imports
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import parallel_run

# modules
def produce_param_list(params: dict, property_list: list, property: str) -> list[dict]:
    """
    Produce a list of the dicts for each experiment

    Parameters
    ----------
    params: dict
        base parameters from which we vary e.g
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

def main(
        RUN_TYPE = 0,
        BASE_PARAMS_LOAD = "package/constants/base_params.json"
         ) -> str: 

    if RUN_TYPE == 0:
        #FOR POLARISATION A,B PLOT
        property_varied = "a_attitude"
        property_varied_title = "Attitude Beta parameters, (a,b)"
        property_values_list = np.asarray([0.05, 0.3, 2.0])
    elif RUN_TYPE == 1:
        ###############################
        #FOR HOMOPHILY PLOT
        property_varied = "homophily"
        property_varied_title = "Attribute homophily, $h$"
        property_values_list = np.asarray([0.1, 0.5, 1.0])

    f = open(BASE_PARAMS_LOAD)
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
        case_2["confirmation_bias"] = 18

        case_1["a_attitude"] = 2.0
        case_1["b_attitude"] = 2.0
        case_1["confirmation_bias"] = 10

        params_list = [case_1, case_2, case_3]
    else:
        params_list = produce_param_list(params, property_values_list, property_varied)

    data_list = parallel_run(params_list) 
    createFolder(fileName)

    save_object(data_list, fileName + "/Data", "data_list")
    save_object(params, fileName + "/Data", "base_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_varied_title, fileName + "/Data", "property_varied_title")
    save_object(property_values_list, fileName + "/Data", "property_values_list")

    return fileName
