"""Performs sobol sensitivity analysis on the model. 

Created: 10/10/2022
"""

# imports
from fileinput import filename
import matplotlib.pyplot as plt
from SALib.analyze import sobol
import numpy.typing as npt
from package.resources.utility import (
    load_object,
)
from package.resources.plot import (
    multi_scatter_seperate_total_sensitivity_analysis_plot,prints_SA_matrix
)
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors

def prep_data(
    Si,
    calc_second_order
):

    if calc_second_order:
        total, first, second = Si.to_df()
        total_data_sa, total_yerr = get_data_bar_chart(total)
        first_data_sa, first_yerr = get_data_bar_chart(first)
        second_data_sa, second_yerr = get_data_bar_chart(second)

        return total_data_sa,total_yerr,first_data_sa,first_yerr,second_data_sa, second_yerr
    else:
        total, first = Si.to_df()
        total_data_sa, total_yerr = get_data_bar_chart(total)
        first_data_sa, first_yerr = get_data_bar_chart(first)

        return total_data_sa,total_yerr,first_data_sa,first_yerr




def get_plot_data(
    problem: dict,
    Y_emissions_flow: npt.NDArray,
    Y_mu: npt.NDArray,
    Y_var: npt.NDArray,
    Y_coefficient_of_variance: npt.NDArray,
    Y_emissions_flow_change: npt.NDArray,
    Y_emissions_stock: npt.NDArray,
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
    Y_emissions_flow: npt.NDArray
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
    Y_emissions_flow_change: npt.NDArray
         Change in emissions between start and finish
    Y_emissions_stock: npt.NDArray
         Total emissions stock
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

    Si_emissions_flow , Si_mu , Si_var , Si_coefficient_of_variance, Si_emissions_flow_change, Si_emissions_stock = analyze_results(problem,Y_emissions_flow,Y_mu,Y_var,Y_coefficient_of_variance,Y_emissions_flow_change, Y_emissions_stock,calc_second_order) 


    if calc_second_order:
        total_data_sa_emissions_flow,total_yerr_emissions_flow,first_data_sa_emissions_flow,first_yerr_emissions_flow, second_data_sa_emissions_flow,second_yerr_emissions_flow   =  prep_data(Si_emissions_flow,calc_second_order)
        total_data_sa_mu,total_yerr_mu,first_data_sa_mu,first_yerr_mu, second_data_sa_mu,second_yerr_mu   =  prep_data(Si_mu,calc_second_order)
        total_data_sa_var,total_yerr_var,first_data_sa_var,first_yerr_var, second_data_sa_var,second_yerr_var   =  prep_data(Si_var,calc_second_order)
        total_data_sa_coefficient_of_variance,total_yerr_coefficient_of_variance,first_data_sa_coefficient_of_variance,first_yerr_coefficient_of_variance, second_data_sa_coefficient_of_variance ,second_yerr_coefficient_of_variance   =  prep_data(Si_coefficient_of_variance,calc_second_order)
        total_data_sa_emissions_flow_change,total_yerr_emissions_flow_change,first_data_sa_emissions_flow_change,first_yerr_emissions_flow_change, second_data_sa_emissions_flow_change,second_yerr_emissions_flow_change   =  prep_data(Si_emissions_flow_change,calc_second_order)
        total_data_sa_emissions_stock,total_yerr_emissions_stock,first_data_sa_emissions_stock,first_yerr_emissions_stock, second_data_sa_emissions_stock,second_yerr_emissions_stock   =  prep_data(Si_emissions_stock,calc_second_order)
    
    else:
        total_data_sa_emissions_flow,total_yerr_emissions_flow,first_data_sa_emissions_flow,first_yerr_emissions_flow   =  prep_data(Si_emissions_flow,calc_second_order)
        total_data_sa_mu,total_yerr_mu,first_data_sa_mu,first_yerr_mu   =  prep_data(Si_mu,calc_second_order)
        total_data_sa_var,total_yerr_var,first_data_sa_var,first_yerr_var   =  prep_data(Si_var,calc_second_order)
        total_data_sa_coefficient_of_variance,total_yerr_coefficient_of_variance,first_data_sa_coefficient_of_variance,first_yerr_coefficient_of_variance   =  prep_data(Si_coefficient_of_variance,calc_second_order)
        total_data_sa_emissions_flow_change,total_yerr_emissions_flow_change,first_data_sa_emissions_flow_change,first_yerr_emissions_flow_change   =  prep_data(Si_emissions_flow_change,calc_second_order)
        total_data_sa_emissions_stock,total_yerr_emissions_stock,first_data_sa_emissions_stock,first_yerr_emissions_stock   =  prep_data(Si_emissions_stock,calc_second_order)
        #total_data_sa_,total_yerr_,first_data_sa_,first_yerr_   =  prep_data(Si_)



    data_sa_dict_total = {
        "emissions_flow": {
            "data": total_data_sa_emissions_flow,
            "yerr": total_yerr_emissions_flow,
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
        "emissions_flow_change": {
            "data": total_data_sa_emissions_flow_change,
            "yerr": total_yerr_emissions_flow_change,
        },
        "emissions_stock": {
            "data": total_data_sa_emissions_stock,
            "yerr": total_yerr_emissions_stock,
        },
    }
    data_sa_dict_first = {
        "emissions_flow": {
            "data": first_data_sa_emissions_flow,
            "yerr": first_yerr_emissions_flow,
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
        "emissions_flow_change": {
            "data": first_data_sa_emissions_flow_change,
            "yerr": first_yerr_emissions_flow_change,
        },
        "emissions_stock": {
            "data": first_data_sa_emissions_stock,
            "yerr": first_yerr_emissions_stock,
        },
    }
    
    if calc_second_order:

        data_sa_dict_second = {
            "emissions_flow": {
                "data": second_data_sa_emissions_flow,
                "yerr": second_yerr_emissions_flow,
            },
            "mu": {
                "data": second_data_sa_mu,
                "yerr": second_yerr_mu,
            },
            "var": {
                "data": second_data_sa_var,
                "yerr": second_yerr_var,
            },
            "coefficient_of_variance": {
                "data": second_data_sa_coefficient_of_variance,
                "yerr": second_yerr_coefficient_of_variance,
            },
            "emissions_flow_change": {
                "data": second_data_sa_emissions_flow_change,
                "yerr": second_yerr_emissions_flow_change,
            },
            "emissions_stock": {
                "data": second_data_sa_emissions_stock,
                "yerr": second_yerr_emissions_stock,
            },
        }
        return data_sa_dict_total, data_sa_dict_first, data_sa_dict_second
    else:
        return data_sa_dict_total, data_sa_dict_first

def get_data_bar_chart(Si_df):
    """
    Taken from: https://salib.readthedocs.io/en/latest/_modules/SALib/plotting/bar.html
    Reduce the sobol index dataframe down to just the bits I want for easy plotting of sobol index and its error

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

    Returns
    -------
    data_sa_dict: dict
        the joined dictionary of dictionaries
    """
    #print("data_sa_dict",data_sa_dict.keys())
    #print("plot_dict",plot_dict.keys())

    for i in data_sa_dict.keys():
        for v in plot_dict[i].keys():
            data_sa_dict[i][v] = plot_dict[i][v]
    return data_sa_dict

def analyze_results(
    problem: dict,
    Y_emissions_flow: npt.NDArray,
    Y_mu: npt.NDArray,
    Y_var: npt.NDArray,
    Y_coefficient_of_variance: npt.NDArray,
    Y_emissions_flow_change: npt.NDArray,
    Y_emissions_stock: npt.NDArray,
    calc_second_order: bool,
) -> tuple:
    """
    Perform sobol analysis on simulation results
    """
    
    Si_emissions_flow = sobol.analyze(
        problem,
        Y_emissions_flow,
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
    Si_emissions_flow_change = sobol.analyze(
        problem,
        Y_emissions_flow_change,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )

    Si_emissions_stock = sobol.analyze(
        problem,
        Y_emissions_stock,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )

    return Si_emissions_flow , Si_mu , Si_var , Si_coefficient_of_variance,Si_emissions_flow_change, Si_emissions_stock




def main(
    fileName,
    plot_outputs = ["emissions_flow","var","emissions_flow_change"],
    dpi_save = 1200,
    latex_bool = 0,
    plot_dict = {
        "emissions_flow": {"title": r"$E_F/NM$", "colour": "red", "linestyle": "--"},
        "mu": {"title": r"$\mu_{I}$", "colour": "blue", "linestyle": "-"},
        "var": {"title": r"$\sigma^{2}_{I}$", "colour": "green", "linestyle": "*"},
        "coefficient_of_variance": {"title": r"$\sigma/\mu$","colour": "orange","linestyle": "-.",},
        "emissions_flow_change": {"title": r"$\Delta E/NM$", "colour": "brown", "linestyle": "-*"},
        "emissions_stock": {"title": r"$E_S/NM$", "colour": "black", "linestyle": "--"},
    },
    titles = [
        r"Number of individuals, $N$", 
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
    ) -> None: 


    problem = load_object(fileName + "/Data", "problem")
    Y_emissions_flow = load_object(fileName + "/Data", "Y_emissions_flow")
    Y_mu = load_object(fileName + "/Data", "Y_mu")
    Y_var = load_object(fileName + "/Data", "Y_var")
    Y_coefficient_of_variance = load_object(fileName + "/Data", "Y_coefficient_of_variance")
    Y_emissions_flow_change = load_object(fileName + "/Data", "Y_emissions_flow_change")
    Y_emissions_stock = load_object(fileName + "/Data", "Y_emissions_stock")
    N_samples = load_object(fileName + "/Data","N_samples" )
    calc_second_order = load_object(fileName + "/Data", "calc_second_order")
    
    if calc_second_order:
        data_sa_dict_total, data_sa_dict_first, data_sa_dict_second  = get_plot_data(problem, Y_emissions_flow, Y_mu, Y_var, Y_coefficient_of_variance,Y_emissions_flow_change,Y_emissions_stock, calc_second_order)
        #print("DONE", data_sa_dict_total, data_sa_dict_first, data_sa_dict_second)
        data_sa_dict_second = Merge_dict_SA(data_sa_dict_second, plot_dict)
    else:
        data_sa_dict_total, data_sa_dict_first = get_plot_data(problem, Y_emissions_flow, Y_mu, Y_var, Y_coefficient_of_variance,Y_emissions_flow_change,Y_emissions_stock, calc_second_order)

    data_sa_dict_first = Merge_dict_SA(data_sa_dict_first, plot_dict)
    data_sa_dict_total = Merge_dict_SA(data_sa_dict_total, plot_dict)
    
    ###############################
    #print(data_sa_dict_first, titles)
    multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first,plot_outputs, titles, dpi_save, N_samples, "First", latex_bool = latex_bool)
    multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_total,plot_outputs, titles, dpi_save, N_samples, "Total", latex_bool = latex_bool)
    
    
    if calc_second_order:
        Browns = mcolors.LinearSegmentedColormap.from_list('white_to_brown', ['#FFFFFF', '#A52A2A'], N=256)

        Si_emissions_flow , Si_mu , Si_var , Si_coefficient_of_variance,Si_emissions_flow_change, Si_emissions_stock = analyze_results(problem,Y_emissions_flow, Y_mu, Y_var, Y_coefficient_of_variance,Y_emissions_flow_change,Y_emissions_stock,calc_second_order) 
        
        second_order_data_emissions_flow = [np.asarray(Si_emissions_flow["S2"]),np.asarray(Si_emissions_flow["S2_conf"])]
        second_order_data_var = [np.asarray(Si_var["S2"]),np.asarray(Si_var["S2_conf"])]
        second_order_data_emissions_flow_change = [np.asarray(Si_emissions_flow_change["S2"]),np.asarray(Si_emissions_flow_change["S2_conf"])]
    
        title_list = ["S2","S2_conf"]
        nrows = 1
        ncols = 2

        prints_SA_matrix(fileName, second_order_data_emissions_flow, title_list,get_cmap("Reds"),nrows, ncols, dpi_save,titles,r"$E/NM$", "Emissions")
        prints_SA_matrix(fileName, second_order_data_var,title_list,get_cmap("Blues"),nrows, ncols, dpi_save, titles,r"$\sigma^{2}$", "var")
        prints_SA_matrix(fileName, second_order_data_emissions_flow_change,title_list,Browns,nrows, ncols, dpi_save,titles,r"$\Delta E/NM$", "Emissions_flow_change")

        #prints_SA_matrix(fileName, data_sa_dict_second["emissions_flow"],title_list,get_cmap("Reds"),nrows, ncols, dpi_save,titles,r"$E/NM$", "Emissions")
        #prints_SA_matrix(fileName, data_sa_dict_second["mu"],title_list,get_cmap("Greens"),nrows, ncols, dpi_save, titles,r"$\mu$", "mu")
        #prints_SA_matrix(fileName, data_sa_dict_second["var"],title_list,get_cmap("Blues"),nrows, ncols, dpi_save, titles,r"$\sigma^{2}$", "var")
        #prints_SA_matrix(fileName, data_sa_dict_second["coefficient_of_variance"],title_list,get_cmap("Oranges"),nrows, ncols, dpi_save, titles,r"$\sigma/\mu$", "coefficient_of_var")        
        #prints_SA_matrix(fileName, data_sa_dict_second["emissions_flow_change"],title_list,Browns,nrows, ncols, dpi_save,titles,r"$\Delta E/NM$", "Emissions_flow_change")

    plt.show()

if __name__ == '__main__':
    main(
    fileName = "results/sensitivity_analysis_00_49_43__07_10_2023",
    plot_outputs = ["emissions_flow","var","emissions_flow_change"],
    dpi_save = 1200,
    latex_bool = 0,
    plot_dict = {
        "emissions_flow": {"title": r"$E/NM$", "colour": "red", "linestyle": "--"},
        "mu": {"title": r"$\mu_{I}$", "colour": "blue", "linestyle": "-"},
        "var": {"title": r"$\sigma^{2}_{I}$", "colour": "green", "linestyle": "*"},
        "coefficient_of_variance": {"title": r"$\sigma/\mu$","colour": "orange","linestyle": "-.",},
        "emissions_flow_change": {"title": r"$\Delta E/NM$", "colour": "brown", "linestyle": "-*"},
        "emissions_stock": {"title": r"$E_S/NM$", "colour": "black", "linestyle": "--"},
    },
    titles = [
        r"Number of individuals, $N$", 
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
)