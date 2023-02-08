"""Performs sobol sensitivity analysis on the model. 
[COMPLETE]

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from SALib.analyze import sobol
import numpy.typing as npt
from resources.utility import (
    load_object,
)
from resources.plot import (
    prints_SA_matrix,
    multi_scatter_seperate_total_sensitivity_analysis_plot,

)

def get_plot_data(
    problem: dict,
    Y_emissions: npt.NDArray,
    Y_mu: npt.NDArray,
    Y_var: npt.NDArray,
    Y_coefficient_of_variance: npt.NDArray,
    Y_emissions_change: npt.NDArray,
    calc_second_order: bool,
) -> tuple[dict, dict]:
    """
    Take the input results data from the sensitivity analysis  experiments for the four variables measures and now preform the analysis to give
    the total, first (and second order) sobol index values for each parameter varied. Then get this into a nice format that can easily be plotted
    with error bars.
    Parameters
    ----------
    problem: dict
        Outlines the number of variables to be varied, the names of these variables and the bounds that they take
    Y_emissions: npt.NDArray
        values for the Emissions = total network emissions/(N*M) at the end of the simulation run time. One entry for each
        parameter set tested
    Y_mu: npt.NDArray
         values for the mean Individual identity normalized by N*M ie mu/(N*M) at the end of the simulation run time.
         One entry for each parameter set tested
    Y_var: npt.NDArray
         values for the variance of Individual identity in the network at the end of the simulation run time. One entry
         for each parameter set tested
    Y_coefficient_of_variance: npt.NDArray
         values for the coefficient of variance of Individual identity normalized by N*M ie (sigma/mu)*(N*M) in the network
         at the end of the simulation run time. One entry for each parameter set tested
    calc_second_order: bool
        Whether or not to conduct second order sobol sensitivity analysis, if set to False then only first and total order results will be
        available. Setting to True increases the total number of runs for the sensitivity analysis but allows for the study of interdependancies
        between parameters
    Returns
    -------
    data_sa_dict_total: dict[dict]
        dictionary containing dictionaries each with data regarding the total order sobol analysis results for each output measure
    data_sa_dict_first: dict[dict]
        dictionary containing dictionaries each with data regarding the first order sobol analysis results for each output measure
    """

    Si_emissions , Si_mu , Si_var , Si_coefficient_of_variance, Si_emissions_change = analyze_results(problem,Y_emissions,Y_mu,Y_var,Y_coefficient_of_variance,Y_emissions_change,calc_second_order) 

    #### Bar chart
    if calc_second_order:
        total_emissions, first_emissions, second_emissions = Si_emissions.to_df()
        total_mu, first_mu, second_mu = Si_mu.to_df()
        total_var, first_var, second_var = Si_var.to_df()
        (
            total_coefficient_of_variance,
            first_coefficient_of_variance,
            second_coefficient_of_variance,
        ) = Si_coefficient_of_variance.to_df()
        total_emissions_change, first_emissions_change, second_emissions_change =  Si_emissions_change.to_df()
    else:
        total_emissions, first_emissions = Si_emissions.to_df()
        total_mu, first_mu = Si_mu.to_df()
        total_var, first_var = Si_var.to_df()
        (
            total_coefficient_of_variance,
            first_coefficient_of_variance,
        ) = Si_coefficient_of_variance.to_df()
        total_emissions_change, first_emissions_change = Si_emissions_change.to_df()

    total_data_sa_emissions, total_yerr_emissions = get_data_bar_chart(total_emissions)
    total_data_sa_mu, total_yerr_mu = get_data_bar_chart(total_mu)
    total_data_sa_var, total_yerr_var = get_data_bar_chart(total_var)
    (
        total_data_sa_coefficient_of_variance,
        total_yerr_coefficient_of_variance,
    ) = get_data_bar_chart(total_coefficient_of_variance)
    total_data_sa_emissions_change, total_yerr_emissions_change = get_data_bar_chart(total_emissions_change)

    first_data_sa_emissions, first_yerr_emissions = get_data_bar_chart(first_emissions)
    first_data_sa_mu, first_yerr_mu = get_data_bar_chart(first_mu)
    first_data_sa_var, first_yerr_var = get_data_bar_chart(first_var)
    (
        first_data_sa_coefficient_of_variance,
        first_yerr_coefficient_of_variance,
    ) = get_data_bar_chart(first_coefficient_of_variance)
    first_data_sa_emissions_change, first_yerr_emissions_change = get_data_bar_chart(first_emissions_change)

    data_sa_dict_total = {
        "emissions": {
            "data": total_data_sa_emissions,
            "yerr": total_yerr_emissions,
        },
        "mu": {
            "data": total_data_sa_mu,
            "yerr": total_yerr_mu,
        },
        "var": {
            "data": total_data_sa_var,
            "yerr": total_yerr_var,
        },
        "coefficient_of_variance": {
            "data": total_data_sa_coefficient_of_variance,
            "yerr": total_yerr_coefficient_of_variance,
        },
        "emissions_change": {
            "data": total_data_sa_emissions_change,
            "yerr": total_yerr_emissions_change,
        },
    }
    data_sa_dict_first = {
        "emissions": {
            "data": first_data_sa_emissions,
            "yerr": first_yerr_emissions,
        },
        "mu": {
            "data": first_data_sa_mu,
            "yerr": first_yerr_mu,
        },
        "var": {
            "data": first_data_sa_var,
            "yerr": first_yerr_var,
        },
        "coefficient_of_variance": {
            "data": first_data_sa_coefficient_of_variance,
            "yerr": first_yerr_coefficient_of_variance,
        },
        "emissions_change": {
            "data": first_data_sa_emissions_change,
            "yerr": first_yerr_emissions_change,
        },
    }

    ##### FINISH NEED TO GENERATE DATA FOR SECOND ORDER ANALYSIS!!

    return data_sa_dict_total, data_sa_dict_first

def get_data_bar_chart(Si_df):
    """
    Taken from: https://salib.readthedocs.io/en/latest/_modules/SALib/plotting/bar.html
    Reduce the sobol index dataframe down to just the bits I want for easy plotting of sobol index and it error
    Parameters
    ----------
    Si_df: pd.DataFrame,
        Dataframe of sensitivity results.
    Returns
    -------
    Sis: pd.Series
        the value of the index
    confs: pd.Series
        the associated error with index
    """

    # magic string indicating DF columns holding conf bound values
    conf_cols = Si_df.columns.str.contains("_conf")
    confs = Si_df.loc[:, conf_cols]  # select all those that ARE in conf_cols!
    confs.columns = [c.replace("_conf", "") for c in confs.columns]
    Sis = Si_df.loc[:, ~conf_cols]  # select all those that ARENT in conf_cols!

    return Sis, confs

def Merge_dict_SA(data_sa_dict: dict, plot_dict: dict) -> dict:
    """
    Merge the dictionaries used to create the data with the plotting dictionaries for easy of plotting later on so that its drawing from
    just one dictionary. This way I seperate the plotting elements from the data generation allowing easier re-plotting. I think this can be
    done with some form of join but I have not worked out how to so far
    Parameters
    ----------
    data_sa_dict: dict
        Dictionary of dictionaries of data associated with each output measure from the sensitivity analysis for a specific sobol index
    plot_dict: dict
        data structure that contains specifics about how a plot should look for each output measure from the sensitivity analysis
        e.g
        plot_dict = {
            "emissions" : {"title": r"$E/NM$",  "colour": "r", "linestyle": "--"},
            "mu" : {"title": r"$\mu/NM$", "colour": "g", "linestyle": "-"},
            "var" : {"title": r"$\sigma^{2}$", "colour": "k", "linestyle": "*"},
            "coefficient_of_variance" : {"title": r"$\sigma NM/\mu$", "colour": "b","linestyle": "-."},
        }
    Returns
    -------
    data_sa_dict: dict
        the joined dictionary of dictionaries
    """
    #print("data_sa_dict",data_sa_dict)
    #print("plot_dict",plot_dict)
    for i in data_sa_dict.keys():
        for v in plot_dict[i].keys():
            #if v in data_sa_dict:
            data_sa_dict[i][v] = plot_dict[i][v]
            #else:
            #    pass
    return data_sa_dict

def analyze_results(
    problem: dict,
    Y_emissions: npt.NDArray,
    Y_mu: npt.NDArray,
    Y_var: npt.NDArray,
    Y_coefficient_of_variance: npt.NDArray,
    Y_emissions_change: npt.NDArray,
    calc_second_order: bool,
) -> tuple:
    """
    Perform sobol analysis on simulation results
    """
    
    Si_emissions = sobol.analyze(
        problem,
        Y_emissions,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )

    Si_mu = sobol.analyze(
        problem, Y_mu, calc_second_order=calc_second_order, print_to_console=False
    )
    Si_var = sobol.analyze(
        problem, Y_var, calc_second_order=calc_second_order, print_to_console=False
    )
    Si_coefficient_of_variance = sobol.analyze(
        problem,
        Y_coefficient_of_variance,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )
    Si_emissions_change = sobol.analyze(
        problem,
        Y_emissions_change,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )

    return Si_emissions , Si_mu , Si_var , Si_coefficient_of_variance,Si_emissions_change


def main(
    fileName = "results\SA_AV_reps_5_samples_15360_D_vars_13_N_samples_1024",
    SECOND_ORDER = 0,
    ) -> None: 
    
    dpi_save = 1200

    plot_dict = {
        "emissions": {"title": r"$E/NM$", "colour": "red", "linestyle": "--"},
        "mu": {"title": r"$\mu$", "colour": "blue", "linestyle": "-"},
        "var": {"title": r"$\sigma^{2}_{I}$", "colour": "green", "linestyle": "*"},
        "coefficient_of_variance": {"title": r"$\sigma/\mu$","colour": "orange","linestyle": "-.",},
        "emissions_change": {"title": r"$\Delta E/NM$", "colour": "brown", "linestyle": "-*"},
    },
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
    ],

    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    problem = load_object(fileName + "/Data", "problem")
    Y_emissions = load_object(fileName + "/Data", "Y_emissions")
    Y_mu = load_object(fileName + "/Data", "Y_mu")
    Y_var = load_object(fileName + "/Data", "Y_var")
    Y_coefficient_of_variance = load_object(
        fileName + "/Data", "Y_coefficient_of_variance"
    )
    Y_emissions_change = load_object(fileName + "/Data", "Y_emissions_change")
    N_samples = load_object(fileName + "/Data","N_samples" )
    calc_second_order = load_object(fileName + "/Data", "calc_second_order")

    data_sa_dict_total, data_sa_dict_first = get_plot_data(
        problem, Y_emissions, Y_mu, Y_var, Y_coefficient_of_variance,Y_emissions_change, calc_second_order
    )#here is where mu and var were the worng way round!

    data_sa_dict_total = Merge_dict_SA(data_sa_dict_total, plot_dict)
    data_sa_dict_first = Merge_dict_SA(data_sa_dict_first, plot_dict)
    
    ###############################
    #remove the re-wiring probability
    for i in data_sa_dict_first.keys():
        del data_sa_dict_first[i]["data"]["S1"]['prob_rewire']
        del data_sa_dict_first[i]["yerr"]["S1"]['prob_rewire']

    multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first, titles, dpi_save, N_samples, "First")
    multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_total, titles, dpi_save, N_samples, "Total")

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
