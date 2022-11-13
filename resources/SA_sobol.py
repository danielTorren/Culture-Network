"""Performs sobol sensitivity analysis on the model. 
A module that constructs dictionary of data to be varied then runs the multiple repeated simulations, 
to account for stochastic variability. Then saves data and parameters used to produce that sensitivity 
analysis data. Finally plots the results.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy.typing as npt
from resources.utility import (
    createFolder,
    save_object,
)
from resources.run import parallel_run_sa

# modules
def sa_run(
    N_samples: int,
    base_params: dict,
    variable_parameters_dict: dict,
    calc_second_order: bool,
) -> tuple[int, dict, str, dict, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Generate the list of dictionaries of parameters for different simulation runs, run them and then save the data from the sensitivity analysis.

    Parameters
    ----------
    N_samples: int
        Number of samples taken per parameter, If calc_second_order is False, the Satelli sample give N * (D + 2), (where D is the number of parameter) parameter sets to run the model
        .There are then extra runs per parameter set to account for stochastic variation. If calc_second_order is True, then this is N * (2D + 2) parameter sets.
    base_params: dict
        This is the set of base parameters which act as the default if a given variable is not tested in the sensitivity analysis e.g
            base_params = {
                "total_time": 2000,#200,
                "delta_t": 1.0,#0.05,
                "compression_factor": 10,
                "save_data": True,
                "alpha_change" : 1.0,
                "harsh_data": False,
                "averaging_method": "Arithmetic",
                "phi_lower": 0.001,
                "phi_upper": 0.005,
                "N": 20,
                "M": 5,
                "K": 10,
                "prob_rewire": 0.2,#0.05,
                "set_seed": 1,
                "culture_momentum_real": 100,#5,
                "learning_error_scale": 0.02,
                "discount_factor": 0.8,
                "present_discount_factor": 0.99,
                "inverse_homophily": 0.2,#0.1,#1 is total mixing, 0 is no mixing
                "homophilly_rate" : 1,
                "confirmation_bias": -100,
                "alpha_attitude": 0.1,
                "beta_attitude": 0.1,
                "alpha_threshold": 1,
                "beta_threshold": 1,
            }
            base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])
    variable_parameters_dict: dict[dict]
        These are the parameters to be varied. The data is structured as follows: The key is the name of the property, the value is composed
        of another dictionary which contains itself several properties; "property": the parameter name, "min": the minimum value to be used in
        the sensitivity analysis, "max": the maximum value to be used in the sensitivity analysis, "title": the name to be used when plotting
        sensitivity analysis results. E.g
            variable_parameters_dict = {
                "N":{"property": "N","min":50,"max":200, "title": r"$N$"},
                "M":{"property":"M","min":1,"max": 10, "title": r"$M$"},
                "K":{"property":"K","min":2,"max":30 , "title": r"$K$"},
                "prob_rewire":{"property":"prob_rewire","min":0.0, "max":0.2 , "title": r"$p_r$"},
            }
    calc_second_order: bool
        Whether or not to conduct second order sobol sensitivity analysis, if set to False then only first and total order results will be
        available. Setting to True increases the total number of runs for the sensitivity analysis but allows for the study of interdependancies
        between parameters

    Returns
    -------
    AV_reps: int
        number of repetitions performed to average over stochastic effects
    problem: dict
        Outlines the number of variables to be varied, the names of these variables and the bounds that they take
    fileName: str
        name of file where results may be found
    param_values: npt.NDArray
        the set of parameter values which are tested
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
    """
    ##AVERAGE RUNS
    AV_reps = len(base_params["seed_list"])
    print("Average reps: ", AV_reps)
    problem, fileName, param_values = generate_problem(
        variable_parameters_dict, N_samples, AV_reps, calc_second_order
    )

    params_list_sa = produce_param_list_SA(
        param_values, base_params, variable_parameters_dict
    )

    Y_emissions, Y_mu, Y_var, Y_coefficient_of_variance, Y_emissions_change = parallel_run_sa(
        params_list_sa
    )
    
    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
    save_object(problem, fileName + "/Data", "problem")
    save_object(Y_emissions, fileName + "/Data", "Y_emissions")
    save_object(Y_mu, fileName + "/Data", "Y_mu")
    save_object(Y_var, fileName + "/Data", "Y_var")
    save_object(Y_coefficient_of_variance, fileName + "/Data", "Y_coefficient_of_variance")
    save_object(Y_emissions_change, fileName + "/Data", "Y_emissions_change")

    return (
        AV_reps,
        problem,
        fileName,
        param_values,
        Y_emissions,
        Y_mu,
        Y_var,
        Y_coefficient_of_variance,
        Y_emissions_change
    )


def generate_problem(
    variable_parameters_dict: dict[dict],
    N_samples: int,
    AV_reps: int,
    calc_second_order: bool,
) -> tuple[dict, str, npt.NDArray]:
    """
    Generate the saltelli.sample given an input set of base and variable parameters, generate filename and folder. Satelli sample used
    is 'a popular quasi-random low-discrepancy sequence used to generate uniform samples of parameter space.' - see the SALib documentation

    Parameters
    ----------
    variable_parameters_dict: dict[dict]
        These are the parameters to be varied. The data is structured as follows: The key is the name of the property, the value is composed
        of another dictionary which contains itself several properties; "property": the parameter name, "min": the minimum value to be used in
        the sensitivity analysis, "max": the maximum value to be used in the sensitivity analysis, "title": the name to be used when plotting
        sensitivity analysis results. See sa_run for example data structure.
    N_samples: int
        Number of samples taken per parameter, If calc_second_order is False, the Satelli sample give N * (D + 2), (where D is the number of parameter) parameter sets to run the model
        .There are then extra runs per parameter set to account for stochastic variation. If calc_second_order is True, then this is N * (2D + 2) parameter sets.
    AV_reps: int
        number of repetitions performed to average over stochastic effects
    calc_second_order: bool
        Whether or not to conduct second order sobol sensitivity analysis, if set to False then only first and total order results will be
        available. Setting to True increases the total number of runs for the sensitivity analysis but allows for the study of interdependancies
        between parameters
    Returns
    -------
    problem: dict
        Outlines the number of variables to be varied, the names of these variables and the bounds that they take
    fileName: str
        name of file where results may be found
    param_values: npt.NDArray
        the set of parameter values which are tested in the sensitivity analysis, generated using saltelli.sample - see:
        https://salib.readthedocs.io/en/latest/api/SALib.sample.html#SALib.sample.saltelli.sample for more details
    """
    D_vars = len(variable_parameters_dict)
    
    if calc_second_order:
        samples = N_samples * (2*D_vars + 2)
    else:
        samples = N_samples * (D_vars + 2)

    print("samples: ", samples)

    names_list = [x["property"] for x in variable_parameters_dict.values()]
    bounds_list = [[x["min"], x["max"]] for x in variable_parameters_dict.values()]

    problem = {
        "num_vars": D_vars,
        "names": names_list,
        "bounds": bounds_list,
    }

    ########################################

    fileName = "results/SA_AV_reps_%s_samples_%s_D_vars_%s_N_samples_%s" % (
        str(AV_reps),
        str(samples),
        str(D_vars),
        str(N_samples),
    )
    print("fileName: ", fileName)
    createFolder(fileName)

    # GENERATE PARAMETER VALUES
    param_values = saltelli.sample(
        problem, N_samples, calc_second_order=calc_second_order
    )  # NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters

    return problem, fileName, param_values


def produce_param_list_SA(
    param_values: npt.NDArray, base_params: dict, variable_parameters_dict: dict[dict]
) -> list:
    """
    Generate the list of dictionaries containing informaton for each experiment. We combine the base_params with the specific variation for
    that experiment from param_values and we just use variable_parameters_dict for the property

    Parameters
    ----------
    param_values: npt.NDArray
        the set of parameter values which are tested in the sensitivity analysis, generated using saltelli.sample - see:
        https://salib.readthedocs.io/en/latest/api/SALib.sample.html#SALib.sample.saltelli.sample for more details
    base_params: dict
        This is the set of base parameters which act as the default if a given variable is not tested in the sensitivity analysis.
        See sa_run for example data structure
    variable_parameters_dict: dict[dict]
        These are the parameters to be varied. The data is structured as follows: The key is the name of the property, the value is composed
        of another dictionary which contains itself several properties; "property": the parameter name, "min": the minimum value to be used in
        the sensitivity analysis, "max": the maximum value to be used in the sensitivity analysis, "title": the name to be used when plotting
        sensitivity analysis results. See sa_run for example data structure.

    Returns
    -------
    params_list: list[dict]
        list of parameter dicts, each entry corresponds to one experiment to be tested
    """

    params_list = []
    for i, X in enumerate(param_values):
        base_params_copy = (
            base_params.copy()
        )  # copy it as we dont want the changes from one experiment influencing another
        variable_parameters_dict_toList = list(
            variable_parameters_dict.values()
        )  # turn it too a list so we can loop through it as X is just an array not a dict
        for v in range(len(X)):  # loop through the properties to be changed
            base_params_copy[variable_parameters_dict_toList[v]["property"]] = X[
                v
            ]  # replace the base variable value with the new value for that experiment
        params_list.append(base_params_copy)
    return params_list

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