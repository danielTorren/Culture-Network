from logging import raiseExceptions
from network import Network
from utility import produceName_alt, createFolder, saveObjects, saveData,createFolderSA
import time
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

def generate_data(parameters: dict) -> Network:
    print_simu = False
    
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


def run(parameters: dict, to_save_list: list, params_name: list) -> str:
    ###GENERATE THE DATA TO BE SAVED

    social_network = generate_data(parameters)

    if parameters["save_data"]:
        steps = len(social_network.history_cultural_var)
        start_time = time.time()
        fileName = produceName_alt(params_name)
        dataName = createFolder(fileName)
        saveObjects(social_network, dataName)
        saveData(
            social_network,
            dataName,
            steps,
            parameters["N"],
            parameters["M"],
            to_save_list[0],
            to_save_list[1],
            to_save_list[2],
            to_save_list[3],
        )
        print("File Path:", fileName)
        print(
            "SAVE time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

        return fileName, social_network
    else:
        return 0, social_network

def two_parameter_run(
    params,
    fileName,
    property_col,
    param_col,
    col_list,
    property_row,
    param_row,
    row_list,
):
    
    data_array = []
    data_list = []

    for i in row_list:
        data_col = []
        for j in col_list:
            params[param_row] = i
            params[param_col] = j
            
            #no change in attention
            data_col.append(generate_data(params))
        data_list = data_list + data_col
        data_array.append(data_col)
    
    title_list = []
    for i in range(len(col_list)):
        for v in range(len(row_list)):
            title_list.append(("%s = %s, %s = %s") % (property_col,str(col_list[i]), property_row,str(row_list[v])))

    print(title_list)

    createFolderSA(fileName)
    
    return data_array, data_list, title_list

def parallel_two_parameter_run(
    params,
    fileName,
    property_col,
    param_col,
    col_list,
    property_row,
    param_row,
    row_list,
):
    
    data_array = []
    data_list = []

    for i in row_list:
        data_col = []
        for j in col_list:
            params[param_row] = i
            params[param_col] = j
            
            #no change in attention
            data_col.append(generate_data(params))
        data_list = data_list + data_col
        data_array.append(data_col)
    
    title_list = []
    for i in range(len(col_list)):
        for v in range(len(row_list)):
            title_list.append(("%s = %s, %s = %s") % (property_col,str(col_list[i]), property_row,str(row_list[v])))

    print(title_list)

    createFolderSA(fileName)
    
    return data_array, data_list, title_list


def parallel_run(params_list):
    num_cores = multiprocessing.cpu_count()
    data_parallel = Parallel(n_jobs=num_cores,verbose=10)(delayed(generate_data)(i) for i in params_list)
    return data_parallel

def get_carbon_emissions_result(params):
    data = generate_data(params)
    return data.total_carbon_emissions/(data.N*data.M)

def parallel_run_sa(params_list,results_property):
    num_cores = multiprocessing.cpu_count()
    if results_property == "Carbon Emissions/NM":
        results_parallel_sa = Parallel(n_jobs=num_cores,verbose=10)(delayed(get_carbon_emissions_result)(i) for i in params_list)
    else:
        raiseExceptions("Invalid results property")
    return results_parallel_sa

def average_get_carbon_emissions_result(params):
    Y_list = []
    for v in params["seed_list"]:
        params["set_seed"] = v
        data = generate_data(params)
        Y_list.append(data.total_carbon_emissions/(data.N*data.M))
    return np.mean(Y_list)

def average_get_mean_coifficient_variance_result(params):
    mean_list = []
    coefficient_variance_list = []
    for v in params["seed_list"]:
        params["set_seed"] = v
        data = generate_data(params)
        mean_list.append(data.average_culture)
        coefficient_variance_list.append(data.std_culture/data.average_culture)
    return np.mean(mean_list), np.mean(coefficient_variance_list)

def average_seed_parallel_run_sa(params_list,results_property):
    num_cores = multiprocessing.cpu_count()
    if results_property == "Carbon Emissions/NM":
        results_parallel_sa = Parallel(n_jobs=num_cores,verbose=10)(delayed(average_get_carbon_emissions_result)(i) for i in params_list)
    else:
        raiseExceptions("Invalid results property")
    return results_parallel_sa

def average_seed_parallel_run_mean_coefficient_variance(params_list):
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores,verbose=10)(delayed(average_get_mean_coifficient_variance_result)(i) for i in params_list)#results_mean, results_coefficient_variance
    results_mean, results_coefficient_variance = zip(*results)
    return np.asarray(results_mean), np.asarray(results_coefficient_variance)#results_mean, results_coefficient_variance

def average_seed_parallel_run_mean_coefficient_variance_multi_run_n(params_list,variable_parameters_dict):

    """Due to parralllisation I cant be sure which data corresponds to which so I need to vary the indivdiual parameters in seperate parallel loops"""
    counter = 0
    combined_data = {}

    for i in variable_parameters_dict.keys():
        results_mean, results_coefficient_variance = average_seed_parallel_run_mean_coefficient_variance(params_list[counter:counter + variable_parameters_dict[i]["reps"]])
        counter += variable_parameters_dict[i]["reps"]
        combined_data["%s" % (i)] = {"mean_data": results_mean, "coefficient_variance_data": results_coefficient_variance}

    return combined_data 

def average_seed_run_sa(params_list,results_property):
    results_parallel_sa = [average_get_carbon_emissions_result(i) for i in params_list]
    return results_parallel_sa
