"""Generate data for sensitivity analysis



Created: 10/10/2022
"""

# imports
import json
from package.resources.utility import (
    save_object,
)
from package.resources.run import parallel_run_sa
from package.generating_data.sensitivity_analysis_gen import generate_problem,produce_param_list_SA

# modules

def main(
        N_samples = 1024,
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA_green.json"
         ) -> str: 
    
    calc_second_order = False

    # load base params
    f = open(BASE_PARAMS_LOAD)
    base_params = json.load(f)

    # load variable params
    f_variable_parameters = open(VARIABLE_PARAMS_LOAD)
    variable_parameters_dict = json.load(f_variable_parameters)
    f_variable_parameters.close()

    ##AVERAGE RUNS
    AV_reps = len(base_params["seed_list"])
    print("Average reps: ", AV_reps)
    print("Params varied: ", len(variable_parameters_dict))

    problem, fileName, param_values = generate_problem(
        variable_parameters_dict, N_samples, AV_reps, calc_second_order
    )

    params_list_sa = produce_param_list_SA(
        param_values, base_params, variable_parameters_dict
    )

    Y_emissions, Y_mu, Y_var, Y_coefficient_of_variance, Y_emissions_change = parallel_run_sa(
        params_list_sa
    )
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(params_list_sa, fileName + "/Data", "params_list_sa")
    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
    save_object(problem, fileName + "/Data", "problem")
    save_object(Y_emissions, fileName + "/Data", "Y_emissions")
    save_object(Y_mu, fileName + "/Data", "Y_mu")
    save_object(Y_var, fileName + "/Data", "Y_var")
    save_object(Y_coefficient_of_variance, fileName + "/Data", "Y_coefficient_of_variance")
    save_object(Y_emissions_change, fileName + "/Data", "Y_emissions_change")
    save_object(N_samples , fileName + "/Data","N_samples")
    save_object(calc_second_order, fileName + "/Data","calc_second_order")

    return fileName

if __name__ == '__main__':
    fileName_Figure_6 = main(
    N_samples = 16,
    BASE_PARAMS_LOAD = "package/constants/base_params_add_greens_SA.json",
    VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA_green.json"
)