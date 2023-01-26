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
RUN = 0  # False,True
if not RUN:
    fileName = "results\SA_AV_reps_5_samples_15360_D_vars_13_N_samples_1024"
N_samples = 1024#256#512#64#1024#512
calc_second_order = False
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

"""
{
    "N":{"property": "N","min":100,"max":500, "title": "$N$"},
    "M":{"property":"M","min":1,"max": 40, "title": "$M$"},
    "K":{"property":"K","min":2,"max":80 , "title": "$K$"},
    "prob_rewire": {"property": "prob_rewire","min": 0.0,"max": 1.0,"title": "$p_r$"},
    "culture_momentum_real":{"property":"culture_momentum_real","min":1,"max": 3000, "title": "$T_{\\rho}$"},
    "learning_error_scale":{"property":"learning_error_scale","min":0.0,"max":1 , "title": "$\\epsilon$" },
    "a_attitude":{"property":"a_attitude","min":0.05, "max": 8, "title": "$a$ Attitude"},
    "b_attitude":{"property":"b_attitude","min":0.05, "max":8 , "title": "$b$ Attitude"},
    "a_threshold":{"property":"a_threshold","min":0.05, "max": 8, "title": "$a$ Threshold"},
    "b_threshold":{"property":"b_threshold","min":0.05, "max": 8, "title": "$b$ Threshold"},
    "discount_factor":{"property":"discount_factor","min":0.0, "max":1.0 , "title": "$\\delta$"},
    "homophily": {"property": "homophily","min": 0.0,"max": 1.0,"title": "$h$"},
    "confirmation_bias":{"property":"confirmation_bias","min":-10.0, "max":100 , "title": "$\\theta$"}
}
"""

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
    r"Confirmation bias, $\theta$"
]

"""
label_dict = {
    "N":r"Number of individuals, $N$", 
    "M":r"Number of behaviours, $M$", 
    "K":r"Mean neighbours, $K$",
    "prob_rewire":r"Probability of re-wiring, $p_r$",
    "culture_momentum_real":r"Cultural momentum, $\rho$",
    "learning_error_scale":r"Social learning error standard deviation, $\sigma_{\varepsilon$}$",
    "a_attitude":r"Attitude Beta $a$",
    "b_attitude":r"Attitude Beta $b$",
    "a_threshold":r"Threshold Beta $a$",
    "b_threshold":r"Threshold Beta $b$",
    "discount_factor":r"Discount factor, $\delta$",
    "homophily":r"Attribute homophily, $h$",
    "confirmation_bias":r"Confirmation bias, $\theta$"
}
"""
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
