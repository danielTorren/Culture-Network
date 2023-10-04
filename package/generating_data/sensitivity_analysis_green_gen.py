"""Generate data for sensitivity analysis for green influencers

Created: 10/10/2022
"""

# imports
from package.generating_data.sensitivity_analysis_gen import main

if __name__ == '__main__':
    fileName_Figure_6 = main(
    N_samples = 64,
    BASE_PARAMS_LOAD = "package/constants/base_params_SA.json",
    VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA_green.json",
    calc_second_order = True
)