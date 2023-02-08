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
import matplotlib.pyplot as plt
import json
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap
from resources.plot import (
    live_print_culture_timeseries,
    print_culture_timeseries_vary_array,
    live_print_culture_timeseries_vary,
    live_compare_plot_animate_behaviour_scatter,
    live_compare_animate_culture_network_and_weighting,
    live_compare_animate_weighting_matrix,
    live_compare_animate_behaviour_matrix,
    # live_print_heterogenous_culture_momentum_double,
    live_average_multirun_double_phase_diagram_mean,
    live_average_multirun_double_phase_diagram_mean_alt,
    live_average_multirun_double_phase_diagram_C_of_V_alt,
    live_average_multirun_double_phase_diagram_C_of_V,
    double_phase_diagram,
    double_phase_diagram_using_meanandvariance,
    double_matrix_plot,
    multi_line_matrix_plot,
    multi_line_matrix_plot_difference,
    multi_line_matrix_plot_difference_percentage,
    multi_line_matrix_plot_divide_through,
    double_matrix_plot_ab,
    double_matrix_plot_cluster,
    double_matrix_plot_cluster_ratio,
    double_matrix_plot_cluster_multi,
    double_matrix_plot_cluster_var_multi,
    double_contour_plot_cluster,
    plot_culture_time_series_emissions,
    plot_behaviours_time_series_emissions,
    plot_behaviours_time_series_emissions_and_culture,
    plot_emissions_distance,
    plot_emissions_multi_ab,
    plot_emissions_multi_ab_relative,
    plot_emissions_multi_ab_relative_all,
    plot_emissions_multi_ab_all,
    plot_behaviours_time_series_culture_and_emissions_ab_relative_all,
    plot_compare_emissions_adding_green,
    live_print_culture_timeseries_green,
    plot_emissions_multi_ab_relative_all_add_green,
    bifurcation_plot_add_green,
    plot_emissions_multi_ab_relative_all_two_theta,
    plot_emissions_multi_ab_relative_all_two_theta_reverse,
    plot_emissions_multi_ab_min_max_two_theta_reverse,
    plot_emissions_multi_ab_min_max_two_theta_reverse_add_green,
    plot_emissions_multi_ab_relative_all_add_green_two_theta,
    plot_emissions_no_culture_add_greens,
)
from resources.utility import (
    createFolder,
    generate_vals_variable_parameters_and_norms,
    save_object,
    load_object,
    produce_name_datetime,
    calc_num_clusters_set_bandwidth,
)
from resources.multi_run_2D_param import (
    generate_title_list,
    shot_two_dimensional_param_run,
    load_data_shot,
    av_two_dimensional_param_run,
    load_data_av,
    reshape_results_matricies,
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
import numpy as np


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

#fileName_no_identity = "results/splitting_eco_warriors_multi_set_N_10_49_28__08_01_2023"#this is the NO identity one
#fileName = "results/splitting_eco_warriors_multi_set_N_10_48_11__08_01_2023"#this is the identity one
#fileName = "results/splitting_eco_warriors_multi_17_39_29__07_01_2023"#this is the NO identity one

fileName = "results/splitting_eco_warriors_add_green_culture_compare_21_03_13__07_02_2023"

#"results/splitting_eco_warriors_add_green_culture_compare_21_04_24__07_02_2023"


#"results/splitting_eco_warriors__add_green_10_20_51__02_02_2023"
#"results/splitting_eco_warriors_single_add_greens_17_44_05__01_02_2023"
#"results/splitting_eco_warriors_distance_reps_10_39_19__31_01_2023"
fileName_DISTANCE_SINGLE_TIME_SERIES = "results/splitting_eco_warriors_distance_single_10_52_16__01_02_2023"

fileName_add_greens_theta_one = "results/splitting_eco_warriors__add_green_10_20_51__02_02_2023"
fileName_add_greens_theta_two = "results/splitting_eco_warriors__add_green_10_23_41__02_02_2023"

fileName_add_greens_culture_theta_one = "results/splitting_eco_warriors_add_green_culture_compare_21_03_13__07_02_2023"#stored in this one
fileName_add_greens_culture_theta_two = "results/splitting_eco_warriors_add_green_culture_compare_21_04_24__07_02_2023"

fileName_theta_one = "results/splitting_eco_warriors_distance_reps_10_17_02__02_02_2023"
fileName_theta_two = "results/splitting_eco_warriors_distance_reps_10_15_39__02_02_2023"

fileName_add_greens_no_culture_theta_one = "results/splitting_eco_warriors_add_green_culture_compare_21_03_13__07_02_2023"#includes data on no culture with greens, theta = 20, STORED HERE
fileName_default_no_culture_theta_one = "results/splitting_eco_warriors__add_green_10_20_51__02_02_2023"#includes data with default, conf = 20
fileName_add_greens_no_culture_theta_two = "results/splitting_eco_warriors_add_green_culture_compare_21_04_24__07_02_2023"#includes data on no culture with greens, theta = 5
fileName_default_no_culture_theta_two = "results/splitting_eco_warriors__add_green_10_23_41__02_02_2023"#includes data with default, conf = 5

#"results/splitting_eco_warriors_distance_single_16_18_30__30_01_2023"
#"results/splitting_eco_warriors_distance_reps_10_29_17__31_01_2023"

#"results/splitting_eco_warriors_distance_reps_17_47_40__30_01_2023"

#"results/splitting_eco_warriors_distance_single_17_31_07__30_01_2023"#timer seriess fro distances 

#"results/splitting_eco_warriors_single_time_series_17_43_32__24_01_2023"#TIME SERIES RUN

#"results/twoD_Average_confirmation_bias_M_200_3000_20_70_20_5"#"results/twoD_Average_confirmation_bias_a_attitude_200_3000_20_64_64_5"#"
#"results/twoD_Average_confirmation_bias_a_attitude_200_3000_20_64_64_5"
#"results/twoD_Average_action_observation_I_a_attitude_200_2000_20_64_64_5"
#twoD_Average_M_confirmation_bias_200_2000_20_40_64_5
#twoD_Average_homophily_confirmation_bias_200_2000_20_64_64_5
#twoD_Average_M_confirmation_bias_200_2000_20_10_402_5

# run bools
RUN = 0 # run or load in previously saved data

SINGLE = 0 # determine if you runs single shots or study the averages over multiple runs for each experiment
MULTI_THETA_M = 0
MULTI = 0
SINGLE_TIME_SERIES = 0
DISTANCE_SINGLE_TIME_SERIES = 0
MULTI_A_B = 0
MULTI_A_B_TWO_THETA = 0
MULTI_A_B_and_DISTANCE_SINGLE_TIME_SERIES = 0
ADD_GREENS_SINGLE = 0
ADD_GREENS_MULTI_A_B = 0
ADD_GREENS_MULTI_A_B_TWO_THETA = 0
ADD_GREENS_MULTI_A_B_COMPARE_CULTURE = 0
ADD_GREENS_MULTI_A_B_COMPARE_CULTURE_TWO_THETA = 0
ADD_GREENS_MULTI_A_B_COMPARE_NO_CULTURE_TWO_THETA = 1

multi_line_plot = 0
DUAL_plot = 0



###PLOT STUFF
dpi_save = 1200
round_dec = 2
cmap_weighting = "Reds"
cmap_edge = get_cmap("Greys")

norm_zero_one = Normalize(vmin=0, vmax=1)

if __name__ == "__main__":
    if SINGLE:
        if RUN:
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

        else:
            base_params = load_object(fileName + "/Data", "base_params")
            print("alpha state: ", base_params["alpha_change"])
            variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
            matrix_emissions = load_object(fileName + "/Data", "matrix_emissions")
    elif MULTI_THETA_M:
        if RUN:
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

        else:
            base_params = load_object(fileName + "/Data", "base_params")
            print("alpha state: ", base_params["alpha_change"])
            variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
            matrix_emissions = load_object(fileName + "/Data", "matrix_emissions")
    elif MULTI:#multi run
        if RUN:
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

        else:
            base_params = load_object(fileName + "/Data", "base_params")
            print("alpha state: ", base_params["alpha_change"])
            variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
            matrix_emissions = load_object(fileName + "/Data", "matrix_emissions")
    elif SINGLE_TIME_SERIES:
        if RUN:
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
        else:
            Data_culture = load_object( fileName + "/Data", "Data_culture")
            Data_no_culture = load_object( fileName + "/Data", "Data_no_culture")
            base_params = load_object( fileName + "/Data", "base_params")
    elif DISTANCE_SINGLE_TIME_SERIES:
        if RUN:
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
        else:
            data_list_culture = load_object( fileName + "/Data", "data_list_culture")
            data_list_no_culture  = load_object( fileName + "/Data", "data_list_no_culture")
            base_params = load_object( fileName + "/Data", "base_params")
            init_attitudes_list = load_object(fileName + "/Data", "init_attitudes_list")
    elif MULTI_A_B:
        if RUN:
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
        else:
            emissions_list_culture = load_object( fileName + "/Data", "emissions_list_culture")
            emissions_list_no_culture  = load_object( fileName + "/Data", "emissions_list_no_culture")
            base_params = load_object( fileName + "/Data", "base_params")
            init_attitudes_list = load_object(fileName + "/Data", "init_attitudes_list")
            mean_list = load_object(fileName + "/Data", "mean_list")
            sum_a_b = load_object(fileName + "/Data", "sum_a_b ")
    elif MULTI_A_B_TWO_THETA:
            emissions_list_culture_theta_one = load_object( fileName_theta_one + "/Data", "emissions_list_culture")
            emissions_list_no_culture_theta_one  = load_object( fileName_theta_one + "/Data", "emissions_list_no_culture")
            base_params_theta_one = load_object( fileName_theta_one + "/Data", "base_params")

            emissions_list_culture_theta_two = load_object( fileName_theta_two + "/Data", "emissions_list_culture")
            emissions_list_no_culture_theta_two  = load_object( fileName_theta_two + "/Data", "emissions_list_no_culture")
            base_params_theta_two = load_object( fileName_theta_two + "/Data", "base_params")

            init_attitudes_list_theta_one = load_object(fileName_theta_one + "/Data", "init_attitudes_list")
            mean_list_theta_one = load_object(fileName_theta_one + "/Data", "mean_list")
            sum_a_b_theta_one = load_object(fileName_theta_one + "/Data", "sum_a_b ")
    elif MULTI_A_B_and_DISTANCE_SINGLE_TIME_SERIES:
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
    elif ADD_GREENS_SINGLE:
        if RUN:
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
        else:
            Data_no_greens = load_object( fileName + "/Data", "Data_no_greens")
            Data_add_greens = load_object( fileName + "/Data", "Data_add_greens")
            base_params = load_object( fileName + "/Data", "base_params")
            base_params_add_green = load_object( fileName + "/Data", "base_params_add_green")
    elif ADD_GREENS_MULTI_A_B:
        if RUN:
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
            root = "splitting_eco_warriors_add_green"
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
        else:
            emissions_list_default = load_object( fileName + "/Data", "emissions_list_default")
            emissions_list_add_green  = load_object( fileName + "/Data", "emissions_list_add_green")
            emissions_id_list_individual_default = load_object( fileName + "/Data", "emissions_id_list_individual_default")
            emissions_id_list_individual_add_green  = load_object( fileName + "/Data", "emissions_id_list_individual_add_green")
            base_params = load_object( fileName + "/Data", "base_params")
            init_attitudes_list = load_object(fileName + "/Data", "init_attitudes_list")
            mean_list = load_object(fileName + "/Data", "mean_list")
            sum_a_b = load_object(fileName + "/Data", "sum_a_b")
            green_N = load_object( fileName + "/Data", "green_N")
            green_K = load_object(fileName + "/Data", "green_K")
            emissions_difference_matrix = load_object(fileName + "/Data", "emissions_difference_matrix")
    elif ADD_GREENS_MULTI_A_B_TWO_THETA:
            emissions_list_default_theta_one = load_object( fileName_add_greens_theta_one + "/Data", "emissions_list_default")
            emissions_list_add_green_theta_one  = load_object( fileName_add_greens_theta_one + "/Data", "emissions_list_add_green")
            emissions_id_list_individual_default_theta_one = load_object( fileName_add_greens_theta_one + "/Data", "emissions_id_list_individual_default")
            emissions_id_list_individual_add_green_theta_one  = load_object( fileName_add_greens_theta_one + "/Data", "emissions_id_list_individual_add_green")
            base_params_theta_one = load_object( fileName_add_greens_theta_one + "/Data", "base_params")
            init_attitudes_list_theta_one = load_object(fileName_add_greens_theta_one + "/Data", "init_attitudes_list")
            mean_list_theta_one = load_object(fileName_add_greens_theta_one + "/Data", "mean_list")
            sum_a_b_theta_one = load_object(fileName_add_greens_theta_one + "/Data", "sum_a_b")
            green_N_theta_one = load_object( fileName_add_greens_theta_one + "/Data", "green_N")
            green_K_theta_one = load_object(fileName_add_greens_theta_one + "/Data", "green_K")
            emissions_difference_matrix_theta_one = load_object(fileName_add_greens_theta_one + "/Data", "emissions_difference_matrix")

            emissions_list_default_theta_two = load_object( fileName_add_greens_theta_two + "/Data", "emissions_list_default")
            emissions_list_add_green_theta_two  = load_object( fileName_add_greens_theta_two + "/Data", "emissions_list_add_green")
            emissions_id_list_individual_default_theta_two = load_object( fileName_add_greens_theta_two + "/Data", "emissions_id_list_individual_default")
            emissions_id_list_individual_add_green_theta_two  = load_object( fileName_add_greens_theta_two + "/Data", "emissions_id_list_individual_add_green")
            base_params_theta_two = load_object( fileName_add_greens_theta_two + "/Data", "base_params")
            init_attitudes_list_theta_two = load_object(fileName_add_greens_theta_two + "/Data", "init_attitudes_list")
            mean_list_theta_two = load_object(fileName_add_greens_theta_two + "/Data", "mean_list")
            sum_a_b_theta_two = load_object(fileName_add_greens_theta_two + "/Data", "sum_a_b")
            green_N_theta_two = load_object( fileName_add_greens_theta_two + "/Data", "green_N")
            green_K_theta_two = load_object(fileName_add_greens_theta_two + "/Data", "green_K")
            emissions_difference_matrix_theta_two = load_object(fileName_add_greens_theta_two + "/Data", "emissions_difference_matrix")
    elif ADD_GREENS_MULTI_A_B_COMPARE_CULTURE:
        if RUN:
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
                "seed_list": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],#[1,2,3],#,#
                "set_seed": 1,
                "N": 200,
                "M": 3,
                "K": 22,
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
            base_params_no_culture = base_params.copy()


            # K HAS BEEN ADJUSTED FROM 20 to 22 to account for the increased density from the additional 20 green indviduals
            ###############################################################
            
            mean_list = np.linspace(0.01,0.99, 200)
            sum_a_b = 4#set the degree of polarisation? i think the more polarised the stronger the effect will be

            init_attitudes_list = gen_atttiudes_list(mean_list, sum_a_b)# GET THE LIST
            params_list_culture = []
            base_params["alpha_change"] = 1.0
            for i in init_attitudes_list:
                #print("i",i)
                base_params["a_attitude"] = i[0]
                base_params["b_attitude"] = i[1]
                params_list_culture.append(base_params.copy())

            base_params_no_culture["alpha_change"] = 2.0
            params_list_no_culture  = []
            for i in init_attitudes_list:
                #print("i",i)
                base_params_no_culture["a_attitude"] = i[0]
                base_params_no_culture["b_attitude"] = i[1]
                params_list_no_culture.append(base_params_no_culture.copy())

            #############################################################

            #fileName = produceName(params, params_name)
            root = "splitting_eco_warriors_add_green_culture_compare"
            fileName = produce_name_datetime(root)
            print("fileName:", fileName)

            ##############################################################################
            #NO CULTURE RUN
            emissions_list_no_culture, emissions_id_list_individual_no_culture = multi_stochstic_emissions_run_all_individual(params_list_no_culture)
            #CULTURED RUN
            emissions_list_culture, emissions_id_list_individual_culture = multi_stochstic_emissions_run_all_individual(params_list_culture)

            
            emissions_difference_lists = []
            for i in range(len(mean_list)):
                mean_row = []
                for k in range(base_params["N"]):
                    person_row = []
                    for j in range(len(base_params["seed_list"])):
                        emissions_difference_stochastic = ((emissions_id_list_individual_culture[i,j][k] -  emissions_id_list_individual_no_culture[i,j][k])/emissions_id_list_individual_no_culture[i,j][k])*100
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
            save_object(emissions_list_no_culture, fileName + "/Data", "emissions_list_no_culture")
            save_object(emissions_list_culture, fileName + "/Data", "emissions_list_culture")
            save_object(emissions_id_list_individual_no_culture, fileName + "/Data", "emissions_id_list_individual_no_culture")
            save_object(emissions_id_list_individual_culture, fileName + "/Data", "emissions_id_list_individual_culture")
            save_object(base_params, fileName + "/Data", "base_params")
            save_object(init_attitudes_list,fileName + "/Data", "init_attitudes_list")
            save_object(emissions_difference_matrix,fileName + "/Data", "emissions_difference_matrix")
        else:
            emissions_list_no_culture = load_object( fileName + "/Data", "emissions_list_no_culture")
            emissions_list_culture  = load_object( fileName + "/Data", "emissions_list_culture")
            emissions_id_list_individual_no_culture = load_object( fileName + "/Data", "emissions_id_list_individual_no_culture")
            emissions_id_list_individual_culture  = load_object( fileName + "/Data", "emissions_id_list_individual_culture")
            base_params = load_object( fileName + "/Data", "base_params")
            init_attitudes_list = load_object(fileName + "/Data", "init_attitudes_list")
            mean_list = load_object(fileName + "/Data", "mean_list")
            sum_a_b = load_object(fileName + "/Data", "sum_a_b")
            emissions_difference_matrix = load_object(fileName + "/Data", "emissions_difference_matrix")      
            print("confirmation_bias",base_params["confirmation_bias"])
    elif ADD_GREENS_MULTI_A_B_COMPARE_CULTURE_TWO_THETA:
            emissions_list_no_culture_theta_one = load_object( fileName_add_greens_culture_theta_one + "/Data", "emissions_list_no_culture")
            emissions_list_culture_theta_one  = load_object( fileName_add_greens_culture_theta_one + "/Data", "emissions_list_culture")
            emissions_id_list_individual_no_culture_theta_one = load_object( fileName_add_greens_culture_theta_one + "/Data", "emissions_id_list_individual_no_culture")
            emissions_id_list_individual_culture_theta_one  = load_object( fileName_add_greens_culture_theta_one + "/Data", "emissions_id_list_individual_culture")
            base_params_theta_one = load_object( fileName_add_greens_culture_theta_one + "/Data", "base_params")
            init_attitudes_list_theta_one = load_object(fileName_add_greens_culture_theta_one + "/Data", "init_attitudes_list")
            mean_list_theta_one = load_object(fileName_add_greens_culture_theta_one + "/Data", "mean_list")
            sum_a_b_theta_one = load_object(fileName_add_greens_culture_theta_one + "/Data", "sum_a_b")
            #green_N_theta_one = load_object(fileName_add_greens_culture_theta_one + "/Data", "green_N")
            #green_K_theta_one = load_object(fileName_add_greens_culture_theta_one + "/Data", "green_K")
            #emissions_difference_matrix_theta_one = load_object(fileName_add_greens_culture_theta_one + "/Data", "emissions_difference_matrix")

            emissions_list_no_culture_theta_two = load_object( fileName_add_greens_culture_theta_two + "/Data", "emissions_list_no_culture")
            emissions_list_culture_theta_two  = load_object( fileName_add_greens_culture_theta_two + "/Data", "emissions_list_culture")
            emissions_id_list_individual_no_culture_theta_two = load_object( fileName_add_greens_culture_theta_two + "/Data", "emissions_id_list_individual_no_culture")
            emissions_id_list_individual_culture_theta_two  = load_object( fileName_add_greens_culture_theta_two + "/Data", "emissions_id_list_individual_culture")
            base_params_theta_two = load_object( fileName_add_greens_culture_theta_two + "/Data", "base_params")
            init_attitudes_list_theta_two = load_object(fileName_add_greens_culture_theta_two + "/Data", "init_attitudes_list")
            mean_list_theta_two = load_object(fileName_add_greens_culture_theta_two + "/Data", "mean_list")
            sum_a_b_theta_two = load_object(fileName_add_greens_culture_theta_two + "/Data", "sum_a_b")
            #green_N_theta_two = load_object(fileName_add_greens_culture_theta_two + "/Data", "green_N")
            #green_K_theta_two = load_object(fileName_add_greens_culture_theta_two + "/Data", "green_K")
            #emissions_difference_matrix_theta_two = load_object(fileName_add_greens_culture_theta_two + "/Data", "emissions_difference_matrix")
    elif ADD_GREENS_MULTI_A_B_COMPARE_NO_CULTURE_TWO_THETA:
            emissions_list_default_theta_one = load_object( fileName_default_no_culture_theta_one + "/Data", "emissions_list_default")
            emissions_list_add_green_theta_one  = load_object( fileName_add_greens_no_culture_theta_one + "/Data", "emissions_list_no_culture")
            emissions_id_list_individual_default_theta_one = load_object( fileName_default_no_culture_theta_one + "/Data", "emissions_id_list_individual_default")
            emissions_id_list_individual_add_green_theta_one  = load_object( fileName_add_greens_no_culture_theta_one + "/Data", "emissions_id_list_individual_no_culture")
            base_params_theta_one = load_object( fileName_add_greens_no_culture_theta_one + "/Data", "base_params")
            init_attitudes_list_theta_one = load_object(fileName_add_greens_no_culture_theta_one + "/Data", "init_attitudes_list")
            mean_list_theta_one = load_object(fileName_add_greens_no_culture_theta_one + "/Data", "mean_list")

            emissions_list_default_theta_two = load_object( fileName_default_no_culture_theta_two + "/Data", "emissions_list_default")
            emissions_list_add_green_theta_two  = load_object( fileName_add_greens_no_culture_theta_two + "/Data", "emissions_list_no_culture")
            emissions_id_list_individual_default_theta_two = load_object( fileName_default_no_culture_theta_two+ "/Data", "emissions_id_list_individual_default")
            emissions_id_list_individual_add_green_theta_two  = load_object( fileName_add_greens_no_culture_theta_two + "/Data", "emissions_id_list_individual_no_culture")
            base_params_theta_two = load_object( fileName_add_greens_no_culture_theta_two + "/Data", "base_params")
            init_attitudes_list_theta_two = load_object(fileName_add_greens_no_culture_theta_two + "/Data", "init_attitudes_list")
            mean_list_theta_two = load_object(fileName_add_greens_no_culture_theta_two + "/Data", "mean_list")

    if multi_line_plot:
        col_dict = variable_parameters_dict["col"]
        row_dict = variable_parameters_dict["row"]

        #### FOR confimation bias vs attitude polarisation
        index_len_x_matrix = col_dict["reps"]
        max_x_val = col_dict["max"]
        min_x_val = col_dict["min"]
        col_ticks_label = list(range(min_x_val, max_x_val, 10))#[-10,0,10,20,30,40,50,60,70,80,90]#[-10,0,10,20,30,40,50,60]#[col_dict["vals"][x] for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        col_ticks_pos = list(range(min_x_val, max_x_val, 10))#[-10,0,10,20,30,40,50,60,70,80,90]#[int(round(index_len_x_matrix*((x - min_x_val)/(max_x_val- min_x_val)))) for x in col_ticks_label]#[0,30,70,50]#[0,10,20,30,40,50,60,70]#[x for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        print("out col_ticks_pos",col_ticks_pos)

                
        index_len_y_matrix =row_dict["reps"]
        max_y_val = row_dict["max"]
        min_y_val = row_dict["min"]
        row_ticks_label = list(range(min_y_val, max_y_val, 10))#[0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[-10,0,10,20,30,40,50,60]#[col_dict["vals"][x] for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        row_ticks_pos = list(range(min_y_val, max_y_val, 10))#[0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[int(round(index_len_y_matrix*((y - min_y_val)/(max_y_val- min_y_val)))) for y in col_ticks_label]#[0,30,70,50]#[0,10,20,30,40,50,60,70]#[x for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        
        #print("row",row_ticks_pos,row_ticks_label)
        #print("col",col_ticks_pos,col_ticks_label)

        row_label = r"Number of behaviours per agent, M"#r"Eco-warriors count"#r"Number of behaviours per agent, M"
        col_label = r'Confirmation bias, $\theta$'#r'Confirmation bias, $\theta$'
        y_label = r"Final emissions, $E$"#r"Identity variance, $\sigma^2$"
        
        multi_line_matrix_plot(fileName,matrix_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 0, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
        multi_line_matrix_plot(fileName,matrix_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 1, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
        #multi_line_matrix_plot(fileName, Z, col_vals, row_vals,  Y_param, cmap, dpi_save, col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label,col_axis_x, col_label, row_label, y_label)
    #### two D plot of emissions with confimation bias and number of eco warriors
    if DUAL_plot:
        col_dict = variable_parameters_dict["col"]
        row_dict = variable_parameters_dict["row"]

        #### FOR confimation bias vs attitude polarisation
        index_len_x_matrix = col_dict["reps"]
        max_x_val = col_dict["max"]
        min_x_val = col_dict["min"]
        col_ticks_label = list(range(min_x_val, max_x_val, 10))#[-10,0,10,20,30,40,50,60,70,80,90]#[-10,0,10,20,30,40,50,60]#[col_dict["vals"][x] for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        col_ticks_pos = list(range(min_x_val, max_x_val, 10))#[-10,0,10,20,30,40,50,60,70,80,90]#[int(round(index_len_x_matrix*((x - min_x_val)/(max_x_val- min_x_val)))) for x in col_ticks_label]#[0,30,70,50]#[0,10,20,30,40,50,60,70]#[x for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        print("out col_ticks_pos",col_ticks_pos)

                
        index_len_y_matrix =row_dict["reps"]
        max_y_val = row_dict["max"]
        min_y_val = row_dict["min"]
        row_ticks_label = list(range(min_y_val, max_y_val, 10))#[0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[-10,0,10,20,30,40,50,60]#[col_dict["vals"][x] for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        row_ticks_pos = list(range(min_y_val, max_y_val, 10))#[0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[int(round(index_len_y_matrix*((y - min_y_val)/(max_y_val- min_y_val)))) for y in col_ticks_label]#[0,30,70,50]#[0,10,20,30,40,50,60,70]#[x for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        
        #print("row",row_ticks_pos,row_ticks_label)
        #print("col",col_ticks_pos,col_ticks_label)

        row_label = r"Number of behaviours per agent, M"#r"Eco-warriors count"#
        col_label = r'Confirmation bias, $\theta$'#r'Confirmation bias, $\theta$'
        y_label = r"Change in final emissions, $\Delta E$"#r"Identity variance, $\sigma^2$"

        base_params_no_identity = load_object(fileName_no_identity + "/Data", "base_params")
        #print("alpha state no identity: ", base_params_no_identity["alpha_change"])
        variable_parameters_dict_no_identity = load_object(fileName_no_identity + "/Data", "variable_parameters_dict")
        matrix_emissions_no_identity = load_object(fileName_no_identity + "/Data", "matrix_emissions")
        
        #print(type( matrix_emissions))
        #print(matrix_emissions.shape)
        
        difference_emissions_matrix = matrix_emissions - matrix_emissions_no_identity

        difference_emissions_matrix_percentage = ((matrix_emissions - matrix_emissions_no_identity)/matrix_emissions_no_identity)*100
        #print("difference_emissions_matrix_percentage", difference_emissions_matrix_percentage)
        #print(difference_emissions_matrix, difference_emissions_matrix.shape)

        #multi_line_matrix_plot_difference(fileName,difference_emissions_matrix, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 0, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
        #multi_line_matrix_plot_difference(fileName,difference_emissions_matrix, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 1, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label


        #multi_line_matrix_plot_difference_percentage(fileName,difference_emissions_matrix_percentage, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 0, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
        multi_line_matrix_plot_difference_percentage(fileName,difference_emissions_matrix_percentage, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 1, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
        #double_matrix_plot(fileName,difference_emissions_matrix, y_label, "emissions",variable_parameters_dict, get_cmap("plasma"),dpi_save,col_ticks_pos,row_ticks_pos,col_ticks_label,row_ticks_label)
        #double_matrix_plot(fileName,difference_emissions_matrix_percentage, y_label, "emissions_percent",variable_parameters_dict, get_cmap("plasma"),dpi_save,col_ticks_pos,row_ticks_pos,col_ticks_label,row_ticks_label)
    if SINGLE_TIME_SERIES:
        #plot_culture_time_series_emissions(fileName,Data_culture, Data_no_culture, dpi_save)
        #plot_behaviours_time_series_emissions(fileName,Data_culture, Data_no_culture, dpi_save)
        plot_behaviours_time_series_emissions_and_culture(fileName,Data_culture, Data_no_culture, dpi_save)
    if DISTANCE_SINGLE_TIME_SERIES:
        plot_emissions_distance(fileName,data_list_culture, data_list_no_culture,init_attitudes_list, dpi_save)
    if MULTI_A_B:
        #plot_emissions_multi_ab(fileName, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save)
        #plot_emissions_multi_ab_relative(fileName, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save)
    
        #plot_emissions_multi_ab_all(fileName, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save, len(base_params["seed_list"]))
        plot_emissions_multi_ab_relative_all(fileName, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save, len(base_params["seed_list"]))
        print(" MULTI_A_B")
    if MULTI_A_B_TWO_THETA:
        #####################################low theta run
        emissions_array_culture_theta_one = np.asarray(emissions_list_culture_theta_one)
        emissions_array_no_culture_theta_one = np.asarray(emissions_list_no_culture_theta_one)

        emissions_difference_theta_one = ((emissions_array_culture_theta_one -  emissions_array_no_culture_theta_one)/emissions_array_no_culture_theta_one)*100
        ###################################high theta run
        emissions_array_culture_theta_two = np.asarray(emissions_list_culture_theta_two)
        emissions_array_no_culture_theta_two = np.asarray(emissions_list_no_culture_theta_two)

        emissions_difference_theta_two = ((emissions_array_culture_theta_two -  emissions_array_no_culture_theta_two)/emissions_array_no_culture_theta_two)*100
        #plot_emissions_multi_ab_relative_all_two_theta(fileName, emissions_difference_theta_one, emissions_difference_theta_two, base_params_theta_one["confirmation_bias"],base_params_theta_two["confirmation_bias"],mean_list, dpi_save, len(base_params_theta_one["seed_list"]))
        #plot_emissions_multi_ab_relative_all_two_theta_reverse(fileName, emissions_difference_theta_one, emissions_difference_theta_two, base_params_theta_one["confirmation_bias"],base_params_theta_two["confirmation_bias"],mean_list, dpi_save, len(base_params_theta_one["seed_list"]))
        plot_emissions_multi_ab_min_max_two_theta_reverse(fileName_theta_one, emissions_difference_theta_one, emissions_difference_theta_two, base_params_theta_one["confirmation_bias"],base_params_theta_two["confirmation_bias"],mean_list_theta_one, dpi_save, len(base_params_theta_one["seed_list"]))
    if MULTI_A_B_and_DISTANCE_SINGLE_TIME_SERIES:
        plot_behaviours_time_series_culture_and_emissions_ab_relative_all(fileName, data_list_culture, data_list_no_culture, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save, len(base_params["seed_list"]))
    if ADD_GREENS_SINGLE:
        plot_compare_emissions_adding_green(fileName,Data_no_greens,Data_add_greens,dpi_save)
        live_print_culture_timeseries_green(fileName, [Data_no_greens,Data_add_greens], "Green_Influencers", ["Default", "Add Green Influencers"],1, 2, dpi_save,["#4421af","#5ad45a"])
    if ADD_GREENS_MULTI_A_B:
        #plot_emissions_multi_ab_relative_all_add_green(fileName, emissions_list_default, emissions_list_add_green, mean_list, dpi_save, len(base_params["seed_list"]))
        bifurcation_plot_add_green(fileName,emissions_difference_matrix, mean_list[::-1], dpi_save)
    if ADD_GREENS_MULTI_A_B_TWO_THETA:
        plot_emissions_multi_ab_min_max_two_theta_reverse_add_green(fileName_add_greens_theta_one,   emissions_difference_matrix_theta_one,    emissions_difference_matrix_theta_two, base_params_theta_one["confirmation_bias"],base_params_theta_two["confirmation_bias"],   mean_list_theta_one,   dpi_save, len(base_params_theta_one["seed_list"])  )
    if ADD_GREENS_MULTI_A_B_COMPARE_CULTURE:
        plot_emissions_multi_ab_relative_all_add_green(fileName, emissions_list_no_culture, emissions_list_culture, mean_list, dpi_save, len(base_params["seed_list"]))
    if ADD_GREENS_MULTI_A_B_COMPARE_CULTURE_TWO_THETA:
        plot_emissions_multi_ab_relative_all_add_green_two_theta(fileName_add_greens_culture_theta_one, emissions_list_no_culture_theta_one, emissions_list_culture_theta_one, emissions_list_no_culture_theta_two, emissions_list_culture_theta_two,base_params_theta_one["confirmation_bias"],base_params_theta_two["confirmation_bias"] ,mean_list_theta_one, dpi_save, len(base_params_theta_one["seed_list"]))
    if ADD_GREENS_MULTI_A_B_COMPARE_NO_CULTURE_TWO_THETA:
        print("SAVED AT: ", fileName_add_greens_no_culture_theta_one)
        plot_emissions_no_culture_add_greens(fileName_add_greens_no_culture_theta_one, emissions_list_default_theta_one, emissions_list_add_green_theta_one, emissions_list_default_theta_two, emissions_list_add_green_theta_two, base_params_theta_one["confirmation_bias"],base_params_theta_two["confirmation_bias"],   mean_list_theta_one,   dpi_save, len(base_params_theta_one["seed_list"])  )

    plt.show()
