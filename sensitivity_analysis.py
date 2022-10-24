"""Performs sobol sensitivity analysis on the model. 
[COMPLETE]

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.cm import get_cmap
from resources.utility import (
    load_object,
)
from resources.plot import (
    prints_SA_matrix,
    bar_sensitivity_analysis_plot,
    scatter_total_sensitivity_analysis_plot,
    multi_scatter_total_sensitivity_analysis_plot,
    multi_scatter_sidebyside_total_sensitivity_analysis_plot,
    multi_scatter_seperate_total_sensitivity_analysis_plot,
)
from resources.SA_sobol import (
    sa_run,
    get_plot_data,
    Merge_dict_SA,
    analyze_results,
)

# constants
RUN = 0  # False,True
fileName = "results/SA_AV_reps_5_samples_112_D_vars_13_N_samples_4"
N_samples = 4
calc_second_order = True

##########################################################################
# plot dict properties
plot_dict = {
    "emissions": {"title": r"$E/NM$", "colour": "r", "linestyle": "--"},
    "mu": {"title": r"$\mu/NM$", "colour": "g", "linestyle": "-"},
    "var": {"title": r"$\sigma^{2}$", "colour": "k", "linestyle": "*"},
    "coefficient_of_variance": {"title": r"$\sigma NM/\mu$","colour": "b","linestyle": "-.",},
}

############################################################################

# Visualize
dpi_save = 1200

if __name__ == "__main__":
    if RUN:
        # load base params
        f = open("constants/base_params.json")
        base_params = json.load(f)
        base_params["time_steps_max"] = int(
            base_params["total_time"] / base_params["delta_t"]
        )

        # load variable params
        f_variable_parameters = open(
            "constants/variable_parameters_dict_SA.json"
        )
        variable_parameters_dict = json.load(f_variable_parameters)
        f_variable_parameters.close()

        (
            AV_reps,
            problem,
            fileName,
            param_values,
            Y_emissions,
            Y_var,
            Y_mu,
            Y_coefficient_of_variance,
        ) = sa_run(N_samples, base_params, variable_parameters_dict, calc_second_order)
    else:
        variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
        problem = load_object(fileName + "/Data", "problem")
        Y_emissions = load_object(fileName + "/Data", "Y_emissions")
        Y_mu = load_object(fileName + "/Data", "Y_mu")
        Y_var = load_object(fileName + "/Data", "Y_var")
        Y_coefficient_of_variance = load_object(
            fileName + "/Data", "Y_coefficient_of_variance"
        )

    data_sa_dict_total, data_sa_dict_first = get_plot_data(
        problem, Y_emissions, Y_var, Y_mu, Y_coefficient_of_variance, calc_second_order
    )

    titles = [x["title"] for x in variable_parameters_dict.values()]

    data_sa_dict_total = Merge_dict_SA(data_sa_dict_total, plot_dict)
    data_sa_dict_first = Merge_dict_SA(data_sa_dict_first, plot_dict)

    ######
    # PLOTS - COMMENT OUT THE ONES YOU DONT WANT

    # multi_scatter_total_sensitivity_analysis_plot(fileName, data_sa_dict_total,titles, dpi_save,N_samples, "Total")
    # multi_scatter_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, titles, dpi_save,N_samples, "First")
    # multi_scatter_sidebyside_total_sensitivity_analysis_plot(fileName, data_sa_dict_total, titles, dpi_save,N_samples, "Total")
    # multi_scatter_sidebyside_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, titles, dpi_save,N_samples, "First")
    #multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, titles, dpi_save, N_samples, "First")

    ####SECOND ORDER
    Si_emissions , Si_mu , Si_var , Si_coefficient_of_variance = analyze_results(problem,Y_emissions,Y_mu,Y_var,Y_coefficient_of_variance,calc_second_order) 
    second_order_data_emissions = [np.asarray(Si_emissions["S2"]),np.asarray(Si_emissions["S2_conf"])]
    second_order_data_mu = [np.asarray(Si_mu["S2"]),np.asarray(Si_mu["S2_conf"])]
    second_order_data_var = [np.asarray(Si_var["S2"]),np.asarray(Si_var["S2_conf"])]
    second_order_data_coefficient_of_variance = [np.asarray(Si_coefficient_of_variance["S2"]),np.asarray(Si_coefficient_of_variance["S2_conf"])]
    title_list = ["S2","S2_conf"]
    nrows = 1
    ncols = 2

    prints_SA_matrix(fileName, second_order_data_emissions,title_list,get_cmap("Reds"),nrows, ncols, dpi_save,titles,r"$E/NM$", "emissions")
    #prints_SA_matrix(fileName, second_order_data_mu,title_list,get_cmap("Greens"),nrows, ncols, dpi_save, titles,r"$\mu/NM$", "mu")
    #prints_SA_matrix(fileName, second_order_data_var,title_list,get_cmap("Blues"),nrows, ncols, dpi_save, titles,r"$\sigma^{2}$", "var")
    #prints_SA_matrix(fileName, second_order_data_coefficient_of_variance,title_list,get_cmap("Oranges"),nrows, ncols, dpi_save, titles,r"$\sigma NM/\mu$", "coefficient_of_var")

    plt.show()
