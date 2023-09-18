"""Performs sobol sensitivity analysis on the model. 

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from SALib.analyze import sobol
import numpy.typing as npt
from package.plotting_data.sensitivity_analysis_plot import get_plot_data, Merge_dict_SA
from package.resources.utility import (
    load_object,
)
from package.resources.plot import (
    multi_scatter_seperate_total_sensitivity_analysis_plot,
)

def main(
    fileName = "results\SA_AV_reps_5_samples_15360_D_vars_13_N_samples_1024",
    plot_outputs = ['emissions','var',"emissions_change"],
    dpi_save = 1200,
    latex_bool = 0
    ) -> None: 
    
    plot_dict = {
        "emissions": {"title": r"$E/NM$", "colour": "red", "linestyle": "--"},
        "mu": {"title": r"$\mu_{I}$", "colour": "blue", "linestyle": "-"},
        "var": {"title": r"$\sigma^{2}_{I}$", "colour": "green", "linestyle": "*"},
        "coefficient_of_variance": {"title": r"$\sigma/\mu$","colour": "orange","linestyle": "-.",},
        "emissions_change": {"title": r"$\Delta E/NM$", "colour": "brown", "linestyle": "-*"},
    }

    titles = [
        r"Number of individuals, $N$", 
        r"Number of green influencers, $N_G$", 
        r"Number of behaviours, $M$", 
        r"Mean neighbours, $K$",
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


    problem = load_object(fileName + "/Data", "problem")
    Y_emissions = load_object(fileName + "/Data", "Y_emissions")
    Y_mu = load_object(fileName + "/Data", "Y_mu")
    Y_var = load_object(fileName + "/Data", "Y_var")
    Y_coefficient_of_variance = load_object(fileName + "/Data", "Y_coefficient_of_variance")
    Y_emissions_change = load_object(fileName + "/Data", "Y_emissions_change")

    N_samples = load_object(fileName + "/Data","N_samples" )
    calc_second_order = load_object(fileName + "/Data", "calc_second_order")

    data_sa_dict_total, data_sa_dict_first = get_plot_data(problem, Y_emissions, Y_mu, Y_var, Y_coefficient_of_variance,Y_emissions_change, calc_second_order)

    data_sa_dict_first = Merge_dict_SA(data_sa_dict_first, plot_dict)
    
    ###############################

    multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first,plot_outputs, titles, dpi_save, N_samples, "First", latex_bool = latex_bool)

    plt.show()

