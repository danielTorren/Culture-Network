from run import parallel_run_sa
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import time
import matplotlib.pyplot as plt
from SALib.plotting.bar import plot as barplot
from SALib.plotting.morris import horizontal_bar_plot
from utility import  createFolderSA,produce_param_list_SA
from plot import (
    prints_SA_matrix
)

# Params
save_data = True
opinion_dynamics =  "DEGROOT" #  "DEGROOT"  "SELECT"
carbon_price_state = False
information_provision_state = False
linear_alpha_diff_state = False#if true use the exponential form instead like theo
homophily_state = True
alpha_change = True
averaging_method = "Arithmetic"#"Threshold weighted arithmetic"

compression_factor = 10

#Social emissions model
#K = 10  # k nearest neighbours INTEGER
#M = 3 # number of behaviours
#N = 100 # number of agents

total_time = 50# THIS REALLY SHOULD NOT BE VARIED AS IT MESSSES UP WITH PREDICTIONS

delta_t = 0.05  # time step size
#culture_momentum_real = 10# real time over which culture is calculated for INTEGER, NEEDS TO BE MROE THAN DELTA t

#prob_rewire = 0.1  # re-wiring probability?

#alpha_attract = 1#2  ##inital distribution parameters - doing the inverse inverts it!
#beta_attract = 1#3
#alpha_threshold = 1#3
#beta_threshold = 1#2

time_steps_max = int(
    total_time / delta_t
)  # number of time steps max, will stop if culture converges
set_seed = 1  ##reproducibility INTEGER
phi_list_lower,phi_list_upper = 0.1,1
#learning_error_scale = 0.02  # 1 standard distribution is 2% error

#inverse_homophily = 0.2#0.2
homophilly_rate = 1

discount_factor = 0.6
present_discount_factor = 0.8

#confirmation_bias = 50

#Infromation provision parameters
if information_provision_state:
    nu = 1# how rapidly extra gains in attractiveness are made
    eta = 0.2#decay rate of information provision boost
    attract_information_provision_list = np.array([0.5*(1/delta_t)]*M)#
    t_IP_matrix = np.array([[],[],[]]) #REAL TIME; list stating at which time steps an information provision policy should be aplied for each behaviour

#Carbon price parameters
if carbon_price_state:
    carbon_price_policy_start = 5#in simualation time to start the policy
    carbon_price_init = 0.0#
    #social_cost_carbon = 0.5
    carbon_price_gradient = 0#social_cost_carbon/time_steps_max# social cost of carbon/total time

params = {
    "opinion_dynamics": opinion_dynamics,
    "save_data": save_data, 
    "compression_factor": compression_factor,
    "time_steps_max": time_steps_max, 
    "carbon_price_state" : carbon_price_state,
    "information_provision_state" : information_provision_state,
    "linear_alpha_diff_state": linear_alpha_diff_state,
    "homophily_state": homophily_state,
    "alpha_change" : alpha_change,
    "averaging_method": averaging_method,
    "delta_t": delta_t,
    "phi_list_lower": phi_list_lower,
    "phi_list_upper": phi_list_upper,
    #"N": N,
    #"M": M,
    #"K": K,
    #"prob_rewire": prob_rewire,
    "set_seed": set_seed,
    #"culture_momentum_real": culture_momentum_real,
    #"learning_error_scale": learning_error_scale,
    #"alpha_attract": alpha_attract,
    #"beta_attract": beta_attract,
    #"alpha_threshold": alpha_threshold,
    #"beta_threshold": beta_threshold,
    "discount_factor": discount_factor,
    #"inverse_homophily": inverse_homophily,#1 is total mixing, 0 is no mixing
    "homophilly_rate" : homophilly_rate,
    "present_discount_factor": present_discount_factor,
    #"confirmation_bias": confirmation_bias,
}

if carbon_price_state:
    params["carbon_price_init"] = carbon_price_init
    params["carbon_price_policy_start"] =  carbon_price_policy_start
    params["carbon_price_gradient"] =  carbon_price_gradient

if information_provision_state:
    params["nu"] = nu
    params["eta"] =  eta
    params["attract_information_provision_list"] = attract_information_provision_list
    params["t_IP_matrix"] =  t_IP_matrix

    ########################################

# variable parameters
variable_parameters = [

    ("N",20,200),
    ("M",1,10), 
    ("K",2,20),
    ("prob_rewire",0.0,0.5),
    #("set_seed",0,10000), 
    ("culture_momentum_real",0.1,50),
    ("learning_error_scale",0.0,0.5), 
    ("inverse_homophily",0.0,1.0), 
    #("discount_factor",0.1,1.0),
    ("confirmation_bias",0.0,100),
    #("homophilly_rate",1,3), 
    #("present_discount_factor",0.1,1),
    #("phi_list_lower",0,0.5),
    #("phi_list_upper",0.5,1),
    ("alpha_attract", 0.1,8),
    ("beta_attract", 0.1,8),
    ("alpha_threshold", 0.1,8),
    ("beta_threshold", 0.1,8),
    ]

results_property = "Carbon Emissions/NM"
N_samples = 16#256#16  # 1024
time_per_step = 0.01045111483335495

if __name__ == "__main__":

    D_vars = len(variable_parameters)
    samples = N_samples * (D_vars + 2)
    print("samples: ",samples)
    names_list = [x[0] for x in variable_parameters]
    bounds_list = [[x[1],x[2]] for x in variable_parameters]

    problem = {
        "num_vars": D_vars,
        "names": names_list,
        "bounds": bounds_list,
    }

    ########################################

    #fileName = produceName_SA(variable_parameters)
    fileName = "results/SA_%s_%s_%s" % (str(samples),str(D_vars),str(N_samples))
    print("fileName: ", fileName)
    createFolderSA(fileName)
    
    #squared_list = [2**x for x in range(10)]
    #time_list_N_samples = [(x, x*(D_vars + 2), (time_per_step * time_steps_max * (x*(D_vars + 2)))/60 ) for x in squared_list]
    #print("time_list_N_samples (N_samples, samples, time in minutes) : ",time_list_N_samples)


    # GENERATE PARAMETER VALUES
    param_values = saltelli.sample(
        problem, N_samples
    )  # NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters

    # run the thing
    start_time = time.time()
    print("start_time =", time.ctime(time.time()))

    params_list_sa = produce_param_list_SA(param_values,params,variable_parameters)
    print("Total runs = ",len(params_list_sa))
    predicted_run_time_seconds = time_per_step * time_steps_max * len(params_list_sa)
    print("Predicted run time = ", predicted_run_time_seconds, "s or ", predicted_run_time_seconds/60, "mins")
    Y = parallel_run_sa(params_list_sa,results_property)#better if a Multiple of 4
    Y = np.asarray(Y)
    """
    Y = np.zeros([param_values.shape[0]])
    for i, X in enumerate(param_values):
        variable_params_added = add_varaiables_to_dict(params,variable_parameters,X)
        social_network = generate_data(variable_params_added)

        ####TOTAL CARBON EMISSIONS PER PERSON PER BEHAVIOUR AS THE FINAL VALUE 
        Y[i] = social_network.total_carbon_emissions/(social_network.N*social_network.M)

        ###CHANGE IN TOTAL EMISSIONS
        #init_value = social_network.history_total_carbon_emissions[0]/(social_network.N*social_network.M)
        #final_value = social_network.history_total_carbon_emissions[-1]/(social_network.N*social_network.M)
        #Y[i] = np.abs(final_value - init_value)/total_time# rate of change of the property
    """
    time_taken = time.time() - start_time
    
    print(
        "RUN time taken: %s minutes" % ((time_taken) / 60), "or %s s" % ((time_taken))
    )

    print("Actual time per step= ", time_taken / (time_steps_max * samples))
    print("results: ",Y)

    ## SAVE RESULTS
    with open(fileName + "/Data_Y.txt", 'w') as f:
        for i in Y:
            f.write(str(i) + '\n')

    Si = sobol.analyze(
        problem, Y,
    )   #print_to_console=True# Si is a Python dict with the keys "S1", "S2", "ST", "S1_conf", "S2_conf", and "ST_conf". The _conf keys store the corresponding confidence intervals, typically with a confidence level of 95%.


    # Visualize
    dpi_save = 2000
    
    BAR_PLOT = 1
    #BAR PLOT
    if BAR_PLOT:
        total, first, second = Si.to_df()
        fig, ax = plt.subplots()
        barplot(total, ax = ax)
        plotName = fileName + "/Plots"
        f = plotName + "/SA.png"
        fig.savefig(f, dpi=dpi_save)

    ####SECOND ORDER
    data = [np.asarray(Si["S2"]),np.asarray(Si["S2_conf"])]
    title_list = ["S2","S2_conf"]
    cmap = "Blues"
    nrows = 1
    ncols = 2

    #print("S2",Si["S2"])
    #print("S2_conf",Si["S2_conf"])
    #print(problem["names"])
    prints_SA_matrix(fileName, data,title_list,cmap,nrows, ncols, dpi_save, problem["names"])

    plt.show()