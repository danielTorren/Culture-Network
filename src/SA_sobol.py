from utility import (
    sa_save_problem,
    sa_save_Y,
    sa_load_problem,
    sa_load_Y,
    createFolderSA,
    produce_param_list_SA
)
from run import parallel_run_sa
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
from plot import (
    prints_SA_matrix,
    bar_sensitivity_analysis_plot,
)
import pandas as pd

base_params = {
    "total_time": 100,
    "delta_t": 0.05,
    "compression_factor": 10,
    "save_data": True, 
    "alpha_change" : 1.0,
    "harsh_data": False,
    "averaging_method": "Arithmetic",
    "phi_list_lower": 0.1,
    "phi_list_upper": 1.0,
    "N": 50,
    "M": 3,
    "K": 20,
    "prob_rewire": 0.05,
    "set_seed": 1,
    "culture_momentum_real": 5,
    "learning_error_scale": 0.02,
    "discount_factor": 0.6,
    "present_discount_factor": 0.8,
    "inverse_homophily": 0.1,#1 is total mixing, 0 is no mixing
    "homophilly_rate" : 1.5,
    "confirmation_bias": 25,
}

base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

#behaviours!
if base_params["harsh_data"]:#trying to create a polarised society!
    base_params["green_extreme_max"]= 8
    base_params["green_extreme_min"]= 2
    base_params["green_extreme_prop"]= 2/5
    base_params["indifferent_max"]= 2
    base_params["indifferent_min"]= 2
    base_params["indifferent_prop"]= 1/5
    base_params["brown_extreme_min"]= 2
    base_params["brown_extreme_max"]= 8
    base_params["brown_extreme_prop"]= 2/5
    if base_params["green_extreme_prop"] + base_params["indifferent_prop"] + base_params["brown_extreme_prop"] != 1:
        raise Exception("Invalid proportions")
else:
    base_params["alpha_attitude"] = 1
    base_params["beta_attitude"] = 1
    base_params["alpha_threshold"] = 1
    base_params["beta_threshold"] = 1

#################################

# variable parameters
variable_parameters = [
    ("M",1,10), 
    ("K",2,30),
    ("N",20,200),
    ("prob_rewire",0.0,0.2),
    ("set_seed",0,10000), 
    ("culture_momentum_real",0.1,50),
    ("learning_error_scale",0.0,0.5), 
    ("alpha_attitude", 0.1,8),
    ("beta_attitude", 0.1,8),
    ("alpha_threshold", 0.1,8),
    ("beta_threshold", 0.1,8),
    ("discount_factor",0.0,1.0),
    ("inverse_homophily",0.0,1.0), 
    ("present_discount_factor",0.0,1.0),
    ("confirmation_bias",0.0,100),
]
"""
Put ones not used here
"""

names = [
    r"$M$",
    r"$K$",
    r"$N$",
    r"$p_r$",
    r"$Seed$",
    r"$T_{\rho}$",
    r"$\eta$",
    r"$Attitude \alpha_{init}$",
    r"$Attitude \beta_{init}$",
    r"$Threshold \alpha_{init}$",
    r"$Threshold \beta_{init}$",
    r"$\delta$",
    r"$h$",
    r"$\beta$",
    r"$\theta$",
]
"""
Put ones not used here
"""

results_property = "Carbon Emissions/NM"
N_samples = 16#256#16  # 1024

# Visualize
dpi_save = 1200

def generate_problem(variable_parameters):

    D_vars = len(variable_parameters)
    samples = N_samples * (2*D_vars + 2)
    print("samples: ",samples)
    names_list = [x[0] for x in variable_parameters]
    bounds_list = [[x[1],x[2]] for x in variable_parameters]

    problem = {
        "num_vars": D_vars,
        "names": names_list,
        "bounds": bounds_list,
    }

    ########################################

    fileName = "results/SA_%s_%s_%s" % (str(samples),str(D_vars),str(N_samples))
    print("fileName: ", fileName)
    createFolderSA(fileName)
    
    # GENERATE PARAMETER VALUES
    param_values = saltelli.sample(
        problem, N_samples
    )  # NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters

    return problem, fileName, param_values

def generate_sa_data(base_params,variable_parameters,param_values):
    params_list_sa = produce_param_list_SA(param_values,base_params,variable_parameters)
    Y = parallel_run_sa(params_list_sa,results_property)
    return np.asarray(Y)

def get_data_bar_chart(Si_df):
    """
        Parameters
    ----------
    * Si_df: pd.DataFrame, of sensitivity results

    TAKEN FROM THE SALIB BAR PLOT: https://salib.readthedocs.io/en/latest/_modules/SALib/plotting/bar.html

    """
    # magic string indicating DF columns holding conf bound values
    conf_cols = Si_df.columns.str.contains('_conf')
    confs = Si_df.loc[:, conf_cols]#select all those that ARE in conf_cols!
    confs.columns = [c.replace('_conf', '') for c in confs.columns]
    Sis = Si_df.loc[:, ~conf_cols]#select all those that ARENT in conf_cols!

    return Sis,confs

if __name__ == "__main__":

    RUN = True#False,True

    if RUN:
        # run the thing
        problem, fileName, param_values = generate_problem(variable_parameters)
        sa_save_problem(problem,fileName)
        Y = generate_sa_data(base_params,variable_parameters,param_values)
        sa_save_Y(Y,fileName)
    else:
        fileName = "results/SA_24_2_4"
        problem = sa_load_problem(fileName)
        Y = sa_load_Y(fileName)
    
    Si = sobol.analyze(problem, Y, print_to_console=False)

    #### Bar chart
    total, first, second = Si.to_df()
    data_sa, yerr = get_data_bar_chart(total)
    bar_sensitivity_analysis_plot(fileName, data_sa, names, yerr , dpi_save)

    #Matrix plot
    data = [np.asarray(Si["S2"]),np.asarray(Si["S2_conf"])]
    title_list = ["S2","S2_conf"]
    cmap = "Blues"
    nrows = 1
    ncols = 2
    prints_SA_matrix(fileName, data,title_list,cmap,nrows, ncols, dpi_save, problem["names"])

    plt.show()