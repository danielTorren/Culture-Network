"""Run multiple simulations varying n parameters
A module that use input data to generate data from multiple social networks varying n properties
between simulations so that the differences may be compared. These multiple runs can either be single 
shot runs or taking the average over multiple runs. Useful for getting a very broad view of the model.
Parameter variation is idenpendant, such that there is a base set of parameters and each of n parameters
is varied from the base line independantly.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# modules
def produceName_multi_run_n(variable_parameters_dict: list, fileName: str) -> str:
    """update original file name that contains the data from the base case with properties that are varied

    Parameters
    ----------
    variable_parameters_dict: dict[dict],
        dictionary of dictionaries  with parameters used to generate attributes, dict used for readability instead of super
        long list of input parameters. Each key in this out dictionary gives the names of the parameter to be varied with details
        of the range and type of distribution of these values found in the value dictionary of each entry.

    fileName: str
        name of file where results may be found composed of value from the different assigned parameters.

    Returns
    -------
    fileName: str
        name of file where results may be found composed of value from the different assigned parameters.
    """

    for i in variable_parameters_dict.keys():
        fileName = fileName + "_" + i
    return fileName


def produce_param_list_n(params, variable_parameters_dict):
    """create a list of the params usign the rep ranges, mins and maxes laid out in variable_parameters_dict. This only varies
     one parameter at a time with the other set to whatever is in the params dict. Each entry corresponds to one experiment

    Parameters
    ----------
    params:dict
        This is the set of base parameters which act as the default if a given variable is not tested in the sensitivity analysis e.g
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

    variable_parameters_dict: dict[dict],
        dictionary of dictionaries  with parameters used to generate attributes, dict used for readability instead of super
        long list of input parameters. Each key in this out dictionary gives the names of the parameter to be varied with details
        of the range and type of distribution of these values found in the value dictionary of each entry. e.g
        variable_parameters_dict = {
            "discount_factor": {"property":"discount_factor","min":-2, "max":0 , "title": r"$\delta$", "divisions": "log", "cmap": get_cmap("Reds"), "cbar_loc": "right", "marker": "o", "reps": 16},
            "inverse_homophily": {"property":"inverse_homophily","min":0.0, "max": 1.0, "title": r"$h$", "divisions": "linear", "cmap": get_cmap("Blues"), "cbar_loc": "right", "marker": "v", "reps": 128},
            "confirmation_bias": {"property":"confirmation_bias","min":0, "max":40, "title": r"$\theta$", "divisions": "linear", "cmap": get_cmap("Greens"), "cbar_loc": "right", "marker": "p", "reps":128},
        }

    Returns
    -------
    params_list: list
        list of parameters for different experiments

    """

    params_list = []
    for i in variable_parameters_dict.values():
        for j in range(i["reps"]):
            params_copy = (
                params.copy()
            )  # Copy it so that i am varying one parameter at a time independently
            params_copy[i["property"]] = i["vals"][j]
            params_list.append(params_copy)
    return params_list