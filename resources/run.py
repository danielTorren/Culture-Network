"""Run simulation 
A module that use input data to run the simulation for a given number of timesteps.
Multiple simulations at once in parallel can also be run. 

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import time
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
import multiprocessing
from resources.network import Network
from resources.utility import calc_num_clusters_auto_bandwidth

# modules
####SINGLE SHOT RUN
def generate_data(parameters: dict) -> Network:
    """
    Generate the Network object which itself contains list of Individual objects. Run this forward in time for the desired number of steps

    Parameters
    ----------
    parameters: dict
        Dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters

        An example of this is:

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

    Returns
    -------
    social_network: Network
        Social network that has evolved from initial conditions
    """

    print_simu = 1  # Whether of not to print how long the single shot simulation took

    if print_simu:
        start_time = time.time()

    social_network = Network(parameters)

    #### RUN TIME STEPS
    time_counter = 0
    while time_counter < parameters["time_steps_max"]:
        social_network.next_step()
        time_counter += 1

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
    return social_network


###MULTIPLE RUNS
def parallel_run(params_dict: dict[dict]) -> list[Network]:
    """
    Generate data from a list of parameter dictionaries, parallelize the execution of each single shot simulation

    Parameters
    ----------
    params_dict: dict[dict],
        dictionary of dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters.
        Each entry corresponds to a different society. See generate_data for an example

    Returns
    -------
    data_parallel: list[list[Network]]
        serialized list of networks, each generated with a different set of parameters
    """

    num_cores = multiprocessing.cpu_count()
    #data_parallel = [generate_data(i) for i in params_dict]
    data_parallel = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_data)(i) for i in params_dict
    )
    return data_parallel


###SENSITIVITY ANALYSIS RUNS
def generate_sensitivity_output(params: dict):
    # -> tuple[float,float,float,float]
    """
    Generate data from a set of parameter contained in a dictionary. Average results over multiple stochastic seeds contained in params["seed_list"]

    Parameters
    ----------
    params: dict,
        dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters. See generate_data for an example

    Returns
    -------
    stochastic_norm_emissions: float
        normalized societal emissions at the end of the allowed time, normalize by number of agents and behaviours per agent. Averaged for different stochastic values
    stochastic_norm_mean: float
        normalized mean societal identity at the end of the allowed time, normalize by number of agents and behaviours per agent and averaged for different stochastic values
    stochastic_norm_var: float
        variance of societal identity at the end of the allowed time, averaged for different stochastic values
    stochastic_norm_coefficient_variance: float
        coefficient of variance (std/mu) of societal identity at the end of the allowed time, averaged for different stochastic values

    """

    emissions_list = []
    mean_list = []
    var_list = []
    coefficient_variance_list = []
    emissions_change_list = []

    norm_factor = params["N"] * params["M"]

    for v in params["seed_list"]:
        params["set_seed"] = v
        data = generate_data(params)

        # Insert more measures below that want to be used for evaluating the
        emissions_list.append(data.total_carbon_emissions / norm_factor)
        mean_list.append(data.average_culture)
        var_list.append(data.var_culture)#data.var_first_behaviour# BOTCH
        coefficient_variance_list.append(data.std_culture / (data.average_culture))
        emissions_change_list.append(np.abs(data.history_total_carbon_emissions[-1] - data.history_total_carbon_emissions[0])/norm_factor)

    stochastic_norm_emissions = np.mean(emissions_list)
    stochastic_norm_mean = np.mean(mean_list)
    stochastic_norm_var = np.mean(var_list)
    stochastic_norm_coefficient_variance = np.mean(coefficient_variance_list)
    stochastic_norm_emissions_change = np.mean(emissions_change_list)

    return (
        stochastic_norm_emissions,
        stochastic_norm_mean,
        stochastic_norm_var,
        stochastic_norm_coefficient_variance,
        stochastic_norm_emissions_change
    )

def generate_cluster_output(params: dict,s):

    emissions_list = []
    mean_list = []
    var_list = []
    coefficient_variance_list = []
    emissions_change_list = []
    clusters_count_list = []

    norm_factor = params["N"] * params["M"]

    for v in params["seed_list"]:
        params["set_seed"] = v
        data = generate_data(params)

        # Insert more measures below that want to be used for evaluating the
        emissions_list.append(data.total_carbon_emissions / norm_factor)
        mean_list.append(data.average_culture)
        var_list.append(data.var_culture)#data.var_first_behaviour# BOTCH
        coefficient_variance_list.append(data.std_culture / (data.average_culture))
        emissions_change_list.append(np.abs(data.total_carbon_emissions - data.init_total_carbon_emissions)/norm_factor)
        clusters_count_list.append(calc_num_clusters_auto_bandwidth(data.culture_list, s))

    stochastic_norm_emissions = np.mean(emissions_list)
    stochastic_norm_mean = np.mean(mean_list)
    stochastic_norm_var = np.mean(var_list)
    stochastic_norm_coefficient_variance = np.mean(coefficient_variance_list)
    stochastic_norm_emissions_change = np.mean(emissions_change_list)
    stochastic_norm_clusters_count = np.mean(clusters_count_list)

    return (
        stochastic_norm_emissions,
        stochastic_norm_mean,
        stochastic_norm_var,
        stochastic_norm_coefficient_variance,
        stochastic_norm_emissions_change,
        stochastic_norm_clusters_count,
    )

def generate_culture_lists_output(params):

    cultures_list = []

    for v in params["seed_list"]:
        params["set_seed"] = v
        data = generate_data(params)
        cultures_list.append(data.culture_list)

    return cultures_list

def culture_data_run(
        params_dict: list[dict]
) -> npt.NDArray:
    #print("params_dict", params_dict)
    num_cores = multiprocessing.cpu_count()
    #res = [generate_sensitivity_output(i) for i in params_dict]
    results_culture_lists = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_culture_lists_output)(i) for i in params_dict
    )

    return np.asarray(results_culture_lists)#can't run with multiple different network sizes
    
def cluster_data_run(
        params_dict: dict[dict],s
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:

    #print("params_dict", params_dict)
    num_cores = multiprocessing.cpu_count()
    #res = [generate_sensitivity_output(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_cluster_output)(i,s) for i in params_dict
    )
    results_emissions, results_mean, results_var, results_coefficient_variance, results_emissions_change, results_clusters_count = zip(
        *res
    )

    return (
        np.asarray(results_emissions),
        np.asarray(results_mean),
        np.asarray(results_var),
        np.asarray(results_coefficient_variance),
        np.asarray(results_emissions_change),
        np.asarray(results_clusters_count)
    )

def parallel_run_sa(
    params_dict: dict[dict],
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Generate data for sensitivity analysis for model varying lots of parameters dictated by params_dict, producing output
    measures emissions,mean,variance and coefficient of variance. Results averaged over multiple runs with different stochastic seed

    Parameters
    ----------
    params_dict: dict[dict],
        dictionary of dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters.
        Each entry corresponds to a different society. See generate_data for an example

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

    #print("params_dict", params_dict)
    num_cores = multiprocessing.cpu_count()
    #res = [generate_sensitivity_output(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_sensitivity_output)(i) for i in params_dict
    )
    results_emissions, results_mean, results_var, results_coefficient_variance, results_emissions_change = zip(
        *res
    )

    return (
        np.asarray(results_emissions),
        np.asarray(results_mean),
        np.asarray(results_var),
        np.asarray(results_coefficient_variance),
        np.asarray(results_emissions_change)
    )


def parallel_run_multi_run_n(params_dict: dict, variable_parameters_dict: dict):

    """
    Generate data for sensitivity analysis for model varying lots of parameters dictated by params_dict, producing output
    measures emissions,mean,variance and coefficient of variance. Results averaged over multiple runs with different stochastic seed.
    Due to parallisation I cant be sure which data corresponds to which so I need to vary the individual parameters in separate parallel loops

    Parameters
    ----------
    params_dict: dict[dict],
        dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters.
        This is the baseline and from this we vary n parameters independantly found in variable_parameters_dict. See generate_data for an example
    variable_parameters_dict:
        dictionary containing the parameters to be varied and extra data for this, each entry represents one variable parameter e.g
        variable_parameters_dict = {
            "discount_factor": {"property":"discount_factor","min":-2, "max":0 , "title": r"$\delta$", "divisions": "log", "cmap": get_cmap("Reds"), "cbar_loc": "right", "marker": "o", "reps": 16},
            "inverse_homophily": {"property":"inverse_homophily","min":0.0, "max": 1.0, "title": r"$h$", "divisions": "linear", "cmap": get_cmap("Blues"), "cbar_loc": "right", "marker": "v", "reps": 128},
        }


    Returns
    -------
    combined_data: dict[dict[str:npt.NDArray]]
        dictionary of dictionaries containing arrays of emissions, identity mean, identity variance and identity coefficient of variance.
        Each dictionary of data corresponds to one parameter being varied ie if i vary confirmation bias and discount parameter my outer dictionary
        would have two entries each dictionaries containing data for the different sensitivity measures.
    """
    counter = 0
    combined_data = {}

    for v in variable_parameters_dict.keys():
        # results = [generate_sensitivity_output(v) for v in params_dict[counter:counter + variable_parameters_dict[i]["reps"]]]
        # print(results)
        # quit()
        (
            results_emissions,
            results_mean,
            results_var,
            results_coefficient_variance,
            results_emissions_change
        ) = parallel_run_sa(
            params_dict[counter : counter + variable_parameters_dict[v]["reps"]]
        )
        counter += variable_parameters_dict[v]["reps"]
        combined_data["%s" % (v)] = {
            "emissions_data": results_emissions,
            "mean_data": results_mean,
            "variance_data": results_var,
            "coefficient_variance_data": results_coefficient_variance,
            "results_emissions_change": results_emissions_change,
        }

    return combined_data
