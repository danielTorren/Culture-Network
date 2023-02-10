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
from model.network import Network

# modules
####SINGLE SHOT RUN
def generate_data(parameters: dict,print_simu = 0) -> Network:
    """
    Generate the Network object which itself contains list of Individual objects. Run this forward in time for the desired number of steps

    Parameters
    ----------
    parameters: dict
        Dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters

    Returns
    -------
    social_network: Network
        Social network that has evolved from initial conditions
    """

    if print_simu:
        start_time = time.time()

    social_network = Network(parameters)

    #### RUN TIME STEPS
    while social_network.t < parameters["time_steps_max"]:
        social_network.next_step()

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
    return social_network

def generate_first_behaviour_lists_one_seed_output(params):
    """For birfurcation just need attitude of first behaviour"""
    data = generate_data(params)
    return [x.attitudes[0] for x in data.agent_list]

def generate_multi_output_individual_emissions_list(params):
    """Individual specific emission and associated id to compare runs with and without behavioural interdependence"""

    emissions_list = []
    emissions_id_individuals_lists = []
    for v in params["seed_list"]:
        params["set_seed"] = v
        data = generate_data(params)

        emissions_list.append(data.total_carbon_emissions)
        emissions_id_individuals_lists.append({x.id:x.total_carbon_emissions for x in data.agent_list if not x.green_fountain_state})

    return (emissions_list, emissions_id_individuals_lists)

def generate_sensitivity_output(params: dict):
    """
    Generate data from a set of parameter contained in a dictionary. Average results over multiple stochastic seeds contained in params["seed_list"]

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
        var_list.append(data.var_culture)
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


def parallel_run(params_dict: dict[dict]) -> list[Network]:
    """
    Generate data from a list of parameter dictionaries, parallelize the execution of each single shot simulation

    """

    num_cores = multiprocessing.cpu_count()
    #data_parallel = [generate_data(i) for i in params_dict]
    data_parallel = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_data)(i) for i in params_dict
    )
    return data_parallel

def multi_stochstic_emissions_run_all_individual(
        params_dict: list[dict]
) -> npt.NDArray:


    num_cores = multiprocessing.cpu_count()
    #results_carbon_emissions = [generate_single_stochastic_output(i) for i in params_dict]
    res = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_multi_output_individual_emissions_list)(i) for i in params_dict
    )
    results_total_carbon_emissions,results_individual_carbon_emissions_id = zip(
        *res
    )


    return np.asarray(results_total_carbon_emissions),np.asarray(results_individual_carbon_emissions_id)

def one_seed_culture_data_run(
        params_dict: list[dict]
) -> npt.NDArray:

    num_cores = multiprocessing.cpu_count()
    #res = [generate_sensitivity_output(i) for i in params_dict]
    results_culture_lists = Parallel(n_jobs=num_cores, verbose=10)(

        delayed(generate_first_behaviour_lists_one_seed_output)(i) for i in params_dict
    )

    return np.asarray(results_culture_lists)#can't run with multiple different network sizes

def parallel_run_sa(
    params_dict: dict[dict],
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
<<<<<<< HEAD:src/run.py
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

    norm_factor = params["N"]*params["M"]

    for v in params["seed_list"]:
        params["set_seed"] = v
        data = generate_data(params)

        #Insert more measures below that want to be used for evaluating the
        emissions_list.append(data.total_carbon_emissions/norm_factor)
        mean_list.append(data.average_culture)
        var_list.append(data.var_culture)
        coefficient_variance_list.append(data.std_culture/(data.average_culture))

    stochastic_norm_emissions  = np.mean(emissions_list)
    stochastic_norm_mean = np.mean(mean_list)
    stochastic_norm_var = np.mean(var_list)
    stochastic_norm_coefficient_variance = np.mean(coefficient_variance_list)

    return stochastic_norm_emissions, stochastic_norm_mean, stochastic_norm_var, stochastic_norm_coefficient_variance

def parallel_run_sa(params_dict: dict[dict]) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Generate data for sensitivity analysis for model varying lots of parameters dictated by params_dict, producing output 
=======
    Generate data for sensitivity analysis for model varying lots of parameters dictated by params_dict, producing output
>>>>>>> re_write_code:package/resources/run.py
    measures emissions,mean,variance and coefficient of variance. Results averaged over multiple runs with different stochastic seed

    """

    #print("params_dict", params_dict)
    num_cores = multiprocessing.cpu_count()
<<<<<<< HEAD:src/run.py
    res = Parallel(n_jobs=num_cores,verbose=10)(delayed(generate_sensitivity_output)(i) for i in params_dict)
    results_emissions, results_mean, results_var, results_coefficient_variance = zip(*res)

    return np.asarray(results_emissions),np.asarray(results_mean), np.asarray(results_var), np.asarray(results_coefficient_variance)

def parallel_run_multi_run_n(params_dict: dict,variable_parameters_dict: dict):

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
        #results = [generate_sensitivity_output(v) for v in params_dict[counter:counter + variable_parameters_dict[i]["reps"]]]
        #print(results)
        #quit()
        results_emissions, results_mean, results_var, results_coefficient_variance = parallel_run_sa(params_dict[counter:counter + variable_parameters_dict[v]["reps"]])
        counter += variable_parameters_dict[v]["reps"]
        combined_data["%s" % (v)] = {"emissions_data": results_emissions,"mean_data": results_mean, "variance_data": results_var,"coefficient_variance_data": results_coefficient_variance}

    return combined_data 


=======
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
>>>>>>> re_write_code:package/resources/run.py
