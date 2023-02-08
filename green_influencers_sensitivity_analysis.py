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
    multi_scatter_parallel_total_sensitivity_analysis_plot,
)
from resources.SA_sobol import (
    sa_run,
    get_plot_data,
    Merge_dict_SA,
    analyze_results,
)

# constants
RUN = 1  # False,True
if not RUN:
    fileName = "results\SA_AV_reps_5_samples_15360_D_vars_13_N_samples_1024"
N_samples = 32 #1024#256#512#64#1024#512
calc_second_order = True
SECOND_ORDER = 0

##########################################################################
# plot dict properties
plot_dict = {
    "emissions": {"title": r"$E/NM$", "colour": "red", "linestyle": "--"},
    "mu": {"title": r"$\mu$", "colour": "blue", "linestyle": "-"},
    "var": {"title": r"$\sigma^{2}_{I}$", "colour": "green", "linestyle": "*"},
    "coefficient_of_variance": {"title": r"$\sigma/\mu$","colour": "orange","linestyle": "-.",},
    "emissions_change": {"title": r"$\Delta E/NM$", "colour": "brown", "linestyle": "-*"},
}

titles = [
    r"Number of individuals, $N$", 
    r"Number of behaviours, $M$", 
    r"Mean neighbours, $K$",
   # r"Probability of re-wiring, $p_r$",
    r"Cultural inertia, $\rho$",
    r"Social learning error, $ \sigma_{ \varepsilon}$ ",
    r"Initial attitude Beta, $a_A$",
    r"Initial attitude Beta, $b_A$",
    r"Initial threshold Beta, $a_T$",
    r"Initial threshold Beta, $b_T$",
    r"Discount factor, $\delta$",
    r"Attribute homophily, $h$",
    r"Confirmation bias, $\theta$",
    r"Green influencers, $N_G$"
]


############################################################################

# Visualize
dpi_save = 1200

if __name__ == "__main__":
    if RUN:
        # load base params
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
                "seed_list": [1],#[1,2,34,5,6,7,8,9,10],#,#
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

        # load variable params
        f_variable_parameters = open(
            "constants/variable_parameters_dict_SA_green_influencers.json"
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
            Y_emissions_change,
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
        Y_emissions_change = load_object(fileName + "/Data", "Y_emissions_change")

    data_sa_dict_total, data_sa_dict_first = get_plot_data(
        problem, Y_emissions, Y_mu, Y_var, Y_coefficient_of_variance,Y_emissions_change, calc_second_order
    )#here is where mu and var were the worng way round!

    #print([x["title"] for x in variable_parameters_dict.values()])
    #print("titles check order",titles)
    #quit()

    data_sa_dict_total = Merge_dict_SA(data_sa_dict_total, plot_dict)
    data_sa_dict_first = Merge_dict_SA(data_sa_dict_first, plot_dict)

    ######
    # PLOTS - COMMENT OUT THE ONES YOU DONT WANT

    #print("data_dict first", data_sa_dict_first)
    

    ###############################
    #remove the re-wiring probability
    for i in data_sa_dict_first.keys():
        #print(data_sa_dict_first[i]["data"])
        #print(data_sa_dict_first[i]["yerr"]["S1"])
        #print(data_sa_dict_first[i]["data"]["S1"]['prob_rewire'])
        #print(data_sa_dict_first[i]["yerr"]["S1"]['prob_rewire'])
        del data_sa_dict_first[i]["data"]["S1"]['prob_rewire']
        del data_sa_dict_first[i]["yerr"]["S1"]['prob_rewire']

    #print("after data_dict first", data_sa_dict_first)
    print(data_sa_dict_first["emissions"]["data"]["S1"])
    print(data_sa_dict_first["emissions"]["yerr"]["S1"])


    #quit()
    ################

    #data_sa_dict_first[""]
    # multi_scatter_total_sensitivity_analysis_plot(fileName, data_sa_dict_total,titles, dpi_save,N_samples, "Total")
    # multi_scatter_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, titles, dpi_save,N_samples, "First")
    #multi_scatter_sidebyside_total_sensitivity_analysis_plot(fileName, data_sa_dict_total, titles, dpi_save,N_samples, "Total")
    #multi_scatter_sidebyside_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, titles, dpi_save,N_samples, "First")
    multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, titles, dpi_save, N_samples, "First")
    #multi_scatter_parallel_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, titles, dpi_save, N_samples, "First")
    #multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_total, titles, dpi_save, N_samples, "Total")


    if SECOND_ORDER:
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
        prints_SA_matrix(fileName, second_order_data_mu,title_list,get_cmap("Greens"),nrows, ncols, dpi_save, titles,r"$\mu$", "mu")
        prints_SA_matrix(fileName, second_order_data_var,title_list,get_cmap("Blues"),nrows, ncols, dpi_save, titles,r"$\sigma^{2}$", "var")
        prints_SA_matrix(fileName, second_order_data_coefficient_of_variance,title_list,get_cmap("Oranges"),nrows, ncols, dpi_save, titles,r"$\sigma/\mu$", "coefficient_of_var")
        
    plt.show()
