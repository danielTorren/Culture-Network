"""Run multiple simulations varying n parameters
[Complete]

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import json
from resources.utility import (
    createFolder,
    generate_vals_variable_parameters_and_norms,
    save_object,
    load_object,
)
from resources.run import parallel_run_multi_run_n
from resources.plot import (
    live_average_multirun_n_diagram_mean_coefficient_variance,
    live_average_multirun_n_diagram_mean_coefficient_variance_cols,
)
from resources.multi_run_n_param import (
    produceName_multi_run_n,
    produce_param_list_n,
)

# constants
###PLOT STUFF
dpi_save = 1200

RUN = 1

if __name__ == "__main__":

    if RUN:
        # load base params
        f_base_params = open("constants/base_params.json")
        params = json.load(f_base_params)
        f_base_params.close()
        params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

        # load variable params
        f_variable_parameters = open("constants/variable_parameters_dict_n.json")
        variable_parameters_dict = json.load(f_variable_parameters)
        f_variable_parameters.close()

        reps = sum([x["reps"] for x in variable_parameters_dict.values()])

        # AVERAGE OVER MULTIPLE RUNS
        fileName = "results/multi_run_n_%s_%s_%s_%s_%s" % (
            str(params["N"]),
            str(params["time_steps_max"]),
            str(params["K"]),
            str(params["seed_list"]),
            str(reps),
        )

        produceName_multi_run_n(variable_parameters_dict, fileName)
        createFolder(fileName)
        print("fileName: ", fileName)

        ### GENERATE PARAMS
        variable_parameters_dict = generate_vals_variable_parameters_and_norms(
            variable_parameters_dict
        )
        params_list = produce_param_list_n(params, variable_parameters_dict)
        ### GENERATE DATA
        combined_data = parallel_run_multi_run_n(params_list, variable_parameters_dict)

        # save the data and params_list  - data,fileName, objectName

        save_object(
            variable_parameters_dict, fileName + "/Data", "variable_parameters_dict"
        )
        save_object(combined_data, fileName + "/Data", "combined data")

    else:
        fileName = "results/multi_run_n_100_600_20_5_262"
        createFolder(fileName)
        variable_parameters_dict = load_object(
            fileName + "/Data", "variable_parameters_dict"
        )
        variable_parameters_dict = generate_vals_variable_parameters_and_norms(
            variable_parameters_dict
        )

        combined_data = load_object(fileName + "/Data", "combined_data")

    ### PLOTS

    # plot_a = live_average_multirun_n_diagram_mean_coefficient_variance(fileName, mean_data_list,coefficient_variance_data_list ,variable_parameters_dict,dpi_save)
    plot_b = live_average_multirun_n_diagram_mean_coefficient_variance(
        fileName,
        combined_data,
        variable_parameters_dict,
        dpi_save,
    )
    plot_c = live_average_multirun_n_diagram_mean_coefficient_variance_cols(
        fileName, combined_data, variable_parameters_dict, dpi_save
    )
    plt.show()
