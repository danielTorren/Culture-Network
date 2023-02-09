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

# imports
import json
import numpy as np
import numpy.typing as npt
from model.network import Network
from resources.utility import (
    createFolder,
    save_object,
    generate_vals_variable_parameters_and_norms,
    produce_name_datetime,
)
from resources.run import (
    parallel_run, 
)

# modules
def produce_param_list_n_double(
    params_dict: dict, variable_parameters_dict: dict[dict]
) -> list[dict]:
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
        for j in variable_parameters_dict["col"]["vals"]:
            params_dict[variable_parameters_dict["row"]["property"]] = i
            params_dict[variable_parameters_dict["col"]["property"]] = j
            params_list.append(params_dict.copy())

    return params_list

def shot_two_dimensional_param_run(
    fileName: str,
    base_params: dict,
    variable_parameters_dict: dict[dict],
    reps_row: int,
    reps_col: int,
) -> tuple[list[Network], npt.NDArray, list[str]]:
    """Generate results for the case of varying two parameters in a single shot for each experiment, also create folder
    and titles for plots and save data

    Parameters
    ----------
    fileName: str
        where to save it e.g in the results folder in data or plots folder
    base_params: dict
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
    params_dict_list = produce_param_list_n_double(base_params, variable_parameters_dict)

    data_list = parallel_run(params_dict_list)
    data_array = np.reshape(data_list, (reps_row, reps_col))
    
    save_object(base_params, fileName + "/Data", "base_params")
    save_data_shot(fileName, variable_parameters_dict, data_list, data_array)

    return data_list, data_array

def save_data_shot(fileName, variable_parameters_dict, data_list, data_array):
    """save variable_parameters_dict and data from single shot experiments"""
    save_object(
        variable_parameters_dict, fileName + "/Data", "variable_parameters_dict"
    )
    save_object(data_list, fileName + "/Data", "data_list")
    save_object(data_array, fileName + "/Data", "data_array")


def main(
    ) -> str: 

    # load base params
    f_base_params = open("constants/base_params.json")
    base_params = json.load(f_base_params)
    f_base_params.close()
    base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

    # load variable params
    f_variable_parameters = open(
        "constants/variable_parameters_dict_2D.json"
    )
    variable_parameters_dict = json.load(f_variable_parameters)
    f_variable_parameters.close()

    # AVERAGE OVER MULTIPLE RUNS
    variable_parameters_dict = generate_vals_variable_parameters_and_norms(
        variable_parameters_dict
    )

    root = "two_param_sweep_average"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    createFolder(fileName)

    return fileName