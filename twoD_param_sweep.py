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
)
from resources.utility import (
    createFolder,
    generate_vals_variable_parameters_and_norms,
)
from resources.multi_run_2D_param import (
    generate_title_list,
    shot_two_dimensional_param_run,
    load_data_shot,
    av_two_dimensional_param_run,
    load_data_av,
    reshape_results_matricies,
)

# run bools
RUN = 1  # run or load in previously saved data
SINGLE = 0  # determine if you runs single shots or study the averages over multiple runs for each experiment
fileName = "results/average_homophily_confirmation_bias_200_2000_20_64_64_5"

###PLOT STUFF
dpi_save = 1200
round_dec = 2
cmap_weighting = "Reds"
cmap_edge = get_cmap("Greys")
fps = 5
interval = 50
layout = "circular"
node_size = 50
cmap = LinearSegmentedColormap.from_list(
    "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
)
norm_zero_one = Normalize(vmin=0, vmax=1)

if __name__ == "__main__":
    if SINGLE:
        if RUN:
            # load base params
            f_base_params = open("constants/base_params.json")
            params = json.load(f_base_params)
            f_base_params.close()
            params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

            # load variable params
            f_variable_parameters = open(
                "constants/variable_parameters_dict_2D.json"
            )
            variable_parameters_dict = json.load(f_variable_parameters)
            f_variable_parameters.close()

            variable_parameters_dict = generate_vals_variable_parameters_and_norms(
                variable_parameters_dict
            )

            fileName = "results/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                variable_parameters_dict["col"]["property"],
                variable_parameters_dict["row"]["property"],
                str(params["N"]),
                str(params["time_steps_max"]),
                str(params["K"]),
                str(variable_parameters_dict["col"]["min"]),
                str(variable_parameters_dict["col"]["max"]),
                str(variable_parameters_dict["row"]["min"]),
                str(variable_parameters_dict["row"]["max"]),
                variable_parameters_dict["col"]["reps"],
                variable_parameters_dict["row"]["reps"],
            )
            print("fileName: ", fileName)

            title_list = generate_title_list(
                variable_parameters_dict["col"]["title"],
                variable_parameters_dict["col"]["vals"],
                variable_parameters_dict["row"]["title"],
                variable_parameters_dict["row"]["vals"],
                round_dec,
            )
            data_list, data_array = shot_two_dimensional_param_run(
                fileName,
                params,
                variable_parameters_dict,
                variable_parameters_dict["row"]["reps"],
                variable_parameters_dict["col"]["reps"],
            )

        else:
            variable_parameters_dict, data_list, data_array = load_data_shot(fileName)

            title_list = generate_title_list(
                variable_parameters_dict["col"]["title"],
                variable_parameters_dict["col"]["vals"],
                variable_parameters_dict["row"]["title"],
                variable_parameters_dict["row"]["vals"],
                round_dec,
            )

        ### PLOTS FOR SINGLE SHOT RUNS
        live_print_culture_timeseries_vary(
            fileName,
            data_list,
            variable_parameters_dict["row"]["property"],
            variable_parameters_dict["col"]["property"],
            title_list,
            variable_parameters_dict["row"]["reps"],
            variable_parameters_dict["col"]["reps"],
            dpi_save,
        )
        # BROKEN print_culture_timeseries_vary_array(fileName, data_array, param_col,property_col,property_varied_values_col,param_row, property_row,property_varied_values_row,  reps_row, reps_col , dpi_save)

        # ani_b = live_compare_animate_culture_network_and_weighting(fileName,data_list,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec,cmap_edge, reps_col, reps_row,property_col,property_varied_values_col)
        # ani_c = live_compare_animate_weighting_matrix(fileName, data_list,  cmap_weighting, interval, fps, round_dec, cmap_edge, reps_row, reps_col,property_col,property_varied_values_col)
        # ani_d = live_compare_animate_behaviour_matrix(fileName, data_list,  cmap, interval, fps, round_dec, reps_row, reps_col,property_col,property_varied_values_col)

    else:
        if RUN:
            # load base params
            f_base_params = open("constants/base_params.json")
            params = json.load(f_base_params)
            f_base_params.close()
            params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

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

            fileName = "results/twoD_Average_%s_%s_%s_%s_%s_%s_%s_%s" % (
                variable_parameters_dict["col"]["property"],
                variable_parameters_dict["row"]["property"],
                str(params["N"]),
                str(params["time_steps_max"]),
                str(params["K"]),
                str(variable_parameters_dict["col"]["reps"]),
                str(variable_parameters_dict["row"]["reps"]),
                len(params["seed_list"]),
            )
            print("fileName: ", fileName)
            print("variable_parameters_dict",variable_parameters_dict)

            (
                results_emissions,
                results_mu,
                results_var,
                results_coefficient_of_variance,
            ) = av_two_dimensional_param_run(fileName, variable_parameters_dict, params)
        else:
            createFolder(fileName)

            (
                variable_parameters_dict,
                results_emissions,
                results_mu,
                results_var,
                results_coefficient_of_variance,
            ) = load_data_av(fileName)

        ###PLOTS FOR STOCHASTICALLY AVERAGED RUNS
        (
            matrix_emissions,
            matrix_mu,
            matrix_var,
            matrix_coefficient_of_variance,
        ) = reshape_results_matricies(
            results_emissions,
            results_mu,
            results_var,
            results_coefficient_of_variance,
            variable_parameters_dict["row"]["reps"],
            variable_parameters_dict["col"]["reps"],
        )

        double_phase_diagram(fileName, matrix_emissions, r"Total normalised emissions $E/NM$", "emissions",variable_parameters_dict, get_cmap("Reds"),dpi_save)
        double_phase_diagram(fileName, matrix_mu, r"Average identity, $\mu$", "mu",variable_parameters_dict, get_cmap("Blues"),dpi_save)
        double_phase_diagram(fileName, matrix_var, r"Identity variance, $\sigma^2$", "variance",variable_parameters_dict, get_cmap("Greens"),dpi_save)
        double_phase_diagram(fileName, matrix_coefficient_of_variance, r"Identity coefficient of variance, $\sigma/\mu$", "coefficient_of_variance",variable_parameters_dict, get_cmap("Oranges"),dpi_save)

        double_phase_diagram_using_meanandvariance(fileName, matrix_emissions, r"Total normalised emissions, $E/NM$", "emissions",variable_parameters_dict, get_cmap("Reds"),dpi_save)
        double_phase_diagram_using_meanandvariance(fileName,matrix_mu,r"Average identity, $\mu$","mu",variable_parameters_dict,get_cmap("Blues"),dpi_save,)
        double_phase_diagram_using_meanandvariance(fileName, matrix_var, r"Identity variance, $\sigma^2$", "variance",variable_parameters_dict, get_cmap("Greens"),dpi_save)
        double_phase_diagram_using_meanandvariance(fileName,matrix_coefficient_of_variance,r"Identity coefficient of variance, $\sigma/\mu$","coefficient_of_variance",variable_parameters_dict,get_cmap("Oranges"),dpi_save,)

    plt.show()
