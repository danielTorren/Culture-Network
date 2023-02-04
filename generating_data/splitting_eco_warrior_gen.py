"""

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import numpy as np
import json
from resources.utility import (
    createFolder,
    generate_vals_variable_parameters_and_norms,
    save_object,
    load_object,
    produce_name_datetime,
)
from twoD_param_sweep_gen import (
    produce_param_list_n_double,
)
from resources.run import (
    single_stochstic_emissions_run,
    multi_stochstic_emissions_run,
    generate_data,
    parallel_run,
    multi_stochstic_emissions_run_all,
    multi_stochstic_emissions_run_all_individual,
)

def calc_new_K(K,N, N_green):
    new_K = (K*(N + N_green - 1))/(N - 1)
    return int(round(new_K))

def gen_atttiudes_list(mean_list, sum_a_b):
    init_attitudes_list = []
    for i in mean_list:
        a = i*sum_a_b
        b = sum_a_b - a
        init_attitudes_list.append([a,b])
    return init_attitudes_list


def main(RUN_NAME = "SINGLE") -> str: 

    if RUN_NAME == "SINGLE":
        # load base params
        f_base_params = open("constants/base_params.json")
        base_params = json.load(f_base_params)
        f_base_params.close()
        base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

        # load variable params
        variable_parameters_dict = {
            "col":{"property":"confirmation_bias","min":0, "max":100 , "title": r"Confirmation bias, $\\theta$","divisions": "linear", "reps": 100},  
            "row":{"property":"green_N","min":0, "max": 100, "title": "Eco warrior count","divisions": "linear", "reps": 100}, 
        }

        variable_parameters_dict = generate_vals_variable_parameters_and_norms(
            variable_parameters_dict
        )

        root = "splitting_eco_warriors_single"
        fileName = produce_name_datetime(root)
        print("fileName:", fileName)
        #print("fileName: ", fileName)

        params_dict_list = produce_param_list_n_double(base_params, variable_parameters_dict)

        emissions_list = single_stochstic_emissions_run(params_dict_list)

        matrix_emissions = emissions_list.reshape((variable_parameters_dict["row"]["reps"], variable_parameters_dict["col"]["reps"]))

        createFolder(fileName)

        save_object(base_params, fileName + "/Data", "base_params")
        save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
        save_object(matrix_emissions, fileName + "/Data", "matrix_emissions")
    elif RUN_NAME == "MULTI_THETA_M":
        # load base params
        f_base_params = open("constants/base_params.json")
        base_params = json.load(f_base_params)
        f_base_params.close()
        base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])
        base_params["seed_list"] = list(range(5))
        print("seed list: ",base_params["seed_list"])

        base_params["green_N"] = 20 # this is 10%
        # load variable params
        variable_parameters_dict = {
            "col":{"property":"confirmation_bias","min":0, "max":50 , "title": r"Confirmation bias, $\\theta$","divisions": "linear", "reps": 50},  
            "row":{"property":"M","min":1, "max": 11, "title": "M","divisions": "linear", "reps": 10}, 
        }

        variable_parameters_dict = generate_vals_variable_parameters_and_norms(
            variable_parameters_dict
        )

        root = "splitting_eco_warriors_multi_set_N"
        fileName = produce_name_datetime(root)
        print("fileName:", fileName)
        #print("fileName: ", fileName)

        params_dict_list = produce_param_list_n_double(base_params, variable_parameters_dict)

        emissions_list = multi_stochstic_emissions_run(params_dict_list)

        matrix_emissions = emissions_list.reshape((variable_parameters_dict["row"]["reps"], variable_parameters_dict["col"]["reps"]))

        createFolder(fileName)

        save_object(base_params, fileName + "/Data", "base_params")
        save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
        save_object(matrix_emissions, fileName + "/Data", "matrix_emissions")
    elif RUN_NAME == "MULTI":#multi run
        # load base params
        f_base_params = open("constants/base_params.json")
        base_params = json.load(f_base_params)
        f_base_params.close()
        base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])
        base_params["seed_list"] = list(range(5))
        print("seed list: ",base_params["seed_list"])
        # load variable params
        variable_parameters_dict = {
            "col":{"property":"confirmation_bias","min":0, "max":100 , "title": r"Confirmation bias, $\\theta$","divisions": "linear", "reps": 40},  
            "row":{"property":"green_N","min":0, "max": 100, "title": "Eco warrior count","divisions": "linear", "reps": 40}, 
        }

        variable_parameters_dict = generate_vals_variable_parameters_and_norms(
            variable_parameters_dict
        )

        root = "splitting_eco_warriors_multi"
        fileName = produce_name_datetime(root)
        print("fileName:", fileName)
        #print("fileName: ", fileName)

        params_dict_list = produce_param_list_n_double(base_params, variable_parameters_dict)

        emissions_list = multi_stochstic_emissions_run(params_dict_list)

        matrix_emissions = emissions_list.reshape((variable_parameters_dict["row"]["reps"], variable_parameters_dict["col"]["reps"]))

        createFolder(fileName)

        save_object(base_params, fileName + "/Data", "base_params")
        save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
        save_object(matrix_emissions, fileName + "/Data", "matrix_emissions")
    elif RUN_NAME == "SINGLE_TIME_SERIES":
        #f = open("constants/base_params.json")
        base_params = {
            "save_timeseries_data": 1, 
            "degroot_aggregation": 1,
            "network_structure": "small_world",
            "alpha_change" : 1.0,
            "guilty_individuals": 0,
            "moral_licensing": 0,
            "immutable_green_fountains": 1,
            "additional_greens":0,
            "polarisation_test": 0,
            "total_time": 3000,
            "delta_t": 1.0,
            "phi_lower": 0.01,
            "phi_upper": 0.05,
            "compression_factor": 10,
            "seed_list": [1,2,3,4,5],
            "set_seed": 1,
            "N": 200,
            "M": 3,
            "K": 20,
            "prob_rewire": 0.1,
            "culture_momentum_real": 1000,
            "learning_error_scale": 0.02,
            "discount_factor": 0.95,
            "homophily": 0.95,
            "homophilly_rate" : 1,
            "confirmation_bias": 20,
            "a_attitude": 1,
            "b_attitude": 1,
            "a_threshold": 1,
            "b_threshold": 1,
            "action_observation_I": 0.0,
            "action_observation_S": 0.0,
            "green_N": 20,
            "guilty_individual_power": 0
        }
        base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

        #fileName = produceName(params, params_name)
        root = "splitting_eco_warriors_single_time_series"
        fileName = produce_name_datetime(root)
        print("fileName:", fileName)
        ##############################################################################
        #CULTURED RUN
        Data_culture = generate_data(base_params)  # run the simulation
        #NO CULTURE RUN
        base_params["alpha_change"] = 2.0
        Data_no_culture = generate_data(base_params)  # run the simulation

        createFolder(fileName)
        save_object(Data_culture, fileName + "/Data", "Data_culture")
        save_object(Data_no_culture, fileName + "/Data", "Data_no_culture")
        save_object(base_params, fileName + "/Data", "base_params")
    elif RUN_NAME == "DISTANCE_SINGLE_TIME_SERIES":
        #f = open("constants/base_params.json")
        base_params = {
            "save_timeseries_data": 1, 
            "degroot_aggregation": 1,
            "network_structure": "small_world",
            "alpha_change" : 1.0,
            "guilty_individuals": 0,
            "moral_licensing": 0,
            "immutable_green_fountains": 1,
            "polarisation_test": 0,
            "total_time": 3000,
            "delta_t": 1.0,
            "phi_lower": 0.01,
            "phi_upper": 0.05,
            "compression_factor": 10,
            "seed_list": [1,2,3,4,5],
            "set_seed": 1,
            "N": 200,
            "M": 3,
            "K": 20,
            "prob_rewire": 0.1,
            "culture_momentum_real": 1000,
            "learning_error_scale": 0.02,
            "discount_factor": 0.95,
            "homophily": 0.95,
            "homophilly_rate" : 1,
            "confirmation_bias": 20,
            "a_attitude": 1,
            "b_attitude": 1,
            "a_threshold": 1,
            "b_threshold": 1,
            "action_observation_I": 0.0,
            "action_observation_S": 0.0,
            "green_N": 20,
            "guilty_individual_power": 0
        }
        base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

        ###############################################################
        init_attitudes_list = [[2,5],[2,2],[5,2]]

        params_list_culture = []

        for i in init_attitudes_list:
            #print("i",i)
            base_params["a_attitude"] = i[0]
            base_params["b_attitude"] = i[1]
            params_list_culture.append(base_params.copy())

        params_list_no_culture  = []
        base_params["alpha_change"] = 2.0
        for i in init_attitudes_list:
            #print("i",i)
            base_params["a_attitude"] = i[0]
            base_params["b_attitude"] = i[1]
            params_list_no_culture.append(base_params.copy())
        #############################################################

        #fileName = produceName(params, params_name)
        root = "splitting_eco_warriors_distance_single"
        fileName = produce_name_datetime(root)
        print("fileName:", fileName)

        ##############################################################################
        #CULTURED RUN
        data_list_culture = parallel_run(params_list_culture)
        #NO CULTURE RUN
        data_list_no_culture = parallel_run(params_list_no_culture)

        createFolder(fileName)
        save_object(data_list_culture, fileName + "/Data", "data_list_culture")
        save_object(data_list_no_culture, fileName + "/Data", "data_list_no_culture")
        save_object(base_params, fileName + "/Data", "base_params")
        save_object(init_attitudes_list,fileName + "/Data", "init_attitudes_list")
    elif RUN_NAME == "MULTI_A_B":
        #f = open("constants/base_params.json")
        base_params = {
            "save_timeseries_data": 0, 
            "degroot_aggregation": 1,
            "network_structure": "small_world",
            "alpha_change" : 1.0,
            "guilty_individuals": 0,
            "moral_licensing": 0,
            "immutable_green_fountains": 1,
            "additional_greens":1,
            "polarisation_test": 0,
            "total_time": 3000,
            "delta_t": 1.0,
            "phi_lower": 0.01,
            "phi_upper": 0.05,
            "compression_factor": 10,
            "seed_list": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            "set_seed": 1,
            "N": 200,
            "M": 3,
            "K": 20,
            "prob_rewire": 0.1,
            "culture_momentum_real": 1000,
            "learning_error_scale": 0.02,
            "discount_factor": 0.95,
            "homophily": 0.95,
            "homophilly_rate" : 1,
            "confirmation_bias": 5,
            "a_attitude": 1,
            "b_attitude": 1,
            "a_threshold": 1,
            "b_threshold": 1,
            "action_observation_I": 0.0,
            "action_observation_S": 0.0,
            "green_N": 20,
            "guilty_individual_power": 0
        }
        base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

        ###############################################################
        
        mean_list = np.linspace(0.01,0.99, 200)
        sum_a_b = 4#set the degree of polarisation? i think the more polarised the stronger the effect will be

        init_attitudes_list = gen_atttiudes_list(mean_list, sum_a_b)# GET THE LIST

        params_list_culture = []

        for i in init_attitudes_list:
            #print("i",i)
            base_params["a_attitude"] = i[0]
            base_params["b_attitude"] = i[1]
            params_list_culture.append(base_params.copy())

        params_list_no_culture  = []
        base_params["alpha_change"] = 2.0
        for i in init_attitudes_list:
            #print("i",i)
            base_params["a_attitude"] = i[0]
            base_params["b_attitude"] = i[1]
            params_list_no_culture.append(base_params.copy())
        #############################################################

        #fileName = produceName(params, params_name)
        root = "splitting_eco_warriors_distance_reps"
        fileName = produce_name_datetime(root)
        print("fileName:", fileName)

        ##############################################################################
        #CULTURED RUN
        emissions_list_culture = multi_stochstic_emissions_run_all(params_list_culture)
        #data_list_culture = parallel_run(params_list_culture)
        #NO CULTURE RUN
        #data_list_no_culture = parallel_run(params_list_no_culture)
        emissions_list_no_culture = multi_stochstic_emissions_run_all(params_list_no_culture)

        createFolder(fileName)
        save_object(mean_list, fileName + "/Data", "mean_list")
        save_object(sum_a_b , fileName + "/Data", "sum_a_b ")
        save_object(emissions_list_culture, fileName + "/Data", "emissions_list_culture")
        save_object(emissions_list_no_culture, fileName + "/Data", "emissions_list_no_culture")
        save_object(base_params, fileName + "/Data", "base_params")
        save_object(init_attitudes_list,fileName + "/Data", "init_attitudes_list")

        data_list_culture = load_object( fileName_DISTANCE_SINGLE_TIME_SERIES + "/Data", "data_list_culture")
        data_list_no_culture  = load_object( fileName_DISTANCE_SINGLE_TIME_SERIES + "/Data", "data_list_no_culture")
        #base_params = load_object( fileName_DISTANCE_SINGLE_TIME_SERIES + "/Data", "base_params")
        #init_attitudes_list = load_object(fileName_DISTANCE_SINGLE_TIME_SERIES + "/Data", "init_attitudes_list")

        emissions_list_culture = load_object( fileName + "/Data", "emissions_list_culture")
        emissions_list_no_culture  = load_object( fileName + "/Data", "emissions_list_no_culture")
        base_params = load_object( fileName + "/Data", "base_params")
        init_attitudes_list = load_object(fileName + "/Data", "init_attitudes_list")
        mean_list = load_object(fileName + "/Data", "mean_list")
        sum_a_b = load_object(fileName + "/Data", "sum_a_b ")
    elif RUN_NAME == "ADD_GREENS_SINGLE":
        #f = open("constants/base_params.json")
        base_params = {
            "save_timeseries_data": 1, 
            "degroot_aggregation": 1,
            "network_structure": "small_world",
            "alpha_change" : 1.0,
            "guilty_individuals": 0,
            "moral_licensing": 0,
            "immutable_green_fountains": 1,
            "additional_greens":1,
            "polarisation_test": 0,
            "total_time": 3000,
            "delta_t": 1.0,
            "phi_lower": 0.01,
            "phi_upper": 0.05,
            "compression_factor": 10,
            "seed_list": [1,2,3,4,5],
            "set_seed": 1,
            "N": 200,
            "M": 3,
            "K": 20,
            "prob_rewire": 0.1,
            "culture_momentum_real": 1000,
            "learning_error_scale": 0.02,
            "discount_factor": 0.95,
            "homophily": 0.95,
            "homophilly_rate" : 1,
            "confirmation_bias": 20,
            "a_attitude": 1,
            "b_attitude": 1,
            "a_threshold": 1,
            "b_threshold": 1,
            "action_observation_I": 0.0,
            "action_observation_S": 0.0,
            "green_N": 0,
            "guilty_individual_power": 0
        }
        base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

        #CULTURED RUN - No Greens
        
        
        base_params_add_green = base_params.copy()
        green_N = 20
        base_params_add_green["green_N"] = green_N
        #base_params_add_green["N"] = base_params["N"] + green_N
        green_K = calc_new_K(base_params["K"],base_params["N"], green_N)
        base_params_add_green["K"] = green_K
        print("green_K, N",green_K,base_params_add_green["N"], base_params_add_green["green_N"])

        root = "splitting_eco_warriors_single_add_greens"
        fileName = produce_name_datetime(root)
        print("fileName:", fileName)
        ##############################################################################


        Data_list = parallel_run([base_params,base_params_add_green])# run the simulation
        
        Data_add_greens = Data_list[1]  #generate_data(base_params_add_green)#
        Data_no_greens = Data_list[0] #generate_data(base_params)# 
        
        
        #CULTURED RUN - Add Greens
        

        createFolder(fileName)
        save_object(Data_no_greens, fileName + "/Data", "Data_no_greens")
        save_object(Data_add_greens, fileName + "/Data", "Data_add_greens")
        save_object(base_params, fileName + "/Data", "base_params")
        save_object(base_params_add_green, fileName + "/Data", "base_params_add_green")
    elif RUN_NAME == "ADD_GREENS_MULTI_A_B":
        #f = open("constants/base_params.json")
        base_params = {
            "save_timeseries_data": 0, 
            "degroot_aggregation": 1,
            "network_structure": "small_world",
            "alpha_change" : 1.0,
            "guilty_individuals": 0,
            "moral_licensing": 0,
            "immutable_green_fountains": 1,
            "additional_greens":1,
            "polarisation_test": 0,
            "total_time": 3000,
            "delta_t": 1.0,
            "phi_lower": 0.01,
            "phi_upper": 0.05,
            "compression_factor": 10,
            "seed_list": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],#[1,2,3],#
            "set_seed": 1,
            "N": 200,
            "M": 3,
            "K": 20,
            "prob_rewire": 0.1,
            "culture_momentum_real": 1000,
            "learning_error_scale": 0.02,
            "discount_factor": 0.95,
            "homophily": 0.95,
            "homophilly_rate" : 1,
            "confirmation_bias": 20,
            "a_attitude": 1,
            "b_attitude": 1,
            "a_threshold": 1,
            "b_threshold": 1,
            "action_observation_I": 0.0,
            "action_observation_S": 0.0,
            "green_N": 0,
            "guilty_individual_power": 0
        }
        base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])
        base_params_add_green = base_params.copy()

        ###############################################################
        
        mean_list = np.linspace(0.01,0.99, 200)
        sum_a_b = 4#set the degree of polarisation? i think the more polarised the stronger the effect will be

        init_attitudes_list = gen_atttiudes_list(mean_list, sum_a_b)# GET THE LIST

        params_list_default = []

        for i in init_attitudes_list:
            #print("i",i)
            base_params["a_attitude"] = i[0]
            base_params["b_attitude"] = i[1]
            params_list_default.append(base_params.copy())

        green_N = 20
        base_params_add_green["green_N"] = green_N
        #base_params_add_green["N"] = base_params["N"] + green_N
        green_K = calc_new_K(base_params["K"],base_params["N"], green_N)
        base_params_add_green["K"] = green_K
        #print("green_K, N",green_K,base_params_add_green["N"], base_params_add_green["green_N"])

        params_list_add_green  = []
        for i in init_attitudes_list:
            #print("i",i)
            base_params_add_green["a_attitude"] = i[0]
            base_params_add_green["b_attitude"] = i[1]
            params_list_add_green.append(base_params_add_green.copy())

        #############################################################

        #fileName = produceName(params, params_name)
        root = "splitting_eco_warriors_distance_reps"
        fileName = produce_name_datetime(root)
        print("fileName:", fileName)

        ##############################################################################
        #CULTURED RUN
        emissions_list_default, emissions_id_list_individual_default = multi_stochstic_emissions_run_all_individual(params_list_default)
        #print("type(emissions_list_default)",type(emissions_id_list_individual_default))
        #print("emissions_id_list_individual_default.shape",emissions_id_list_individual_default.shape)
        #data_list_culture = parallel_run(params_list_culture)
        #NO CULTURE RUN
        #data_list_no_culture = parallel_run(params_list_no_culture)
        emissions_list_add_green, emissions_id_list_individual_add_green = multi_stochstic_emissions_run_all_individual(params_list_add_green)

        #emissions_list_individual_add_green is a list or matrix? the entries are difference means and then for each stochastic run a list of indiviudal emissions [means, stochastic, indivdual emissions]
        #emissions_pos_matrix needs to be means then indivduals, so need to aggregate across the stochastic runs, for the differences between individuals (they are indifferet places but same initial values)

        #go through each stochastic run and sutract

        emissions_difference_lists = []
        for i in range(len(mean_list)):
            mean_row = []
            for k in range(base_params["N"]):
                person_row = []
                for j in range(len(base_params["seed_list"])):
                    emissions_difference_stochastic = ((emissions_id_list_individual_add_green[i,j][k] -  emissions_id_list_individual_default[i,j][k])/emissions_id_list_individual_default[i,j][k])*100
                    person_row.append(emissions_difference_stochastic)
                mean_row.append(np.mean(person_row))
            emissions_difference_lists.append(mean_row)
        
        emissions_difference_matrix = np.asarray(emissions_difference_lists)
        #print("eemissions_difference_matrix.shape",emissions_difference_matrix,emissions_difference_matrix.shape)

        #emissions_difference_matrix = np.mean(emissions_difference_stochastic, axis=0) 
        #print("emissions_difference_matrix.shape",emissions_difference_matrix.shape)

        createFolder(fileName)
        save_object(mean_list, fileName + "/Data", "mean_list")
        save_object(sum_a_b , fileName + "/Data", "sum_a_b")
        save_object(emissions_list_default, fileName + "/Data", "emissions_list_default")
        save_object(emissions_list_add_green, fileName + "/Data", "emissions_list_add_green")
        save_object(emissions_id_list_individual_default, fileName + "/Data", "emissions_id_list_individual_default")
        save_object(emissions_id_list_individual_add_green, fileName + "/Data", "emissions_id_list_individual_add_green")
        save_object(base_params, fileName + "/Data", "base_params")
        save_object(init_attitudes_list,fileName + "/Data", "init_attitudes_list")
        save_object(green_N, fileName + "/Data", "green_N")
        save_object(green_K, fileName + "/Data", "green_K")
        save_object(emissions_difference_matrix,fileName + "/Data", "emissions_difference_matrix")

    return fileName

