from run import generate_data
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import time
import matplotlib.pyplot as plt
from SALib.plotting.bar import plot as barplot
from SALib.plotting.morris import horizontal_bar_plot
from utility import  produceName_SA,createFolderSA

if __name__ == "__main__":
    N_samples = 32  # 1024
    
    # variable parameters
    variable_parameters = [("phi_list_lower",0,0.7),("phi_list_upper",0.7,1),("N",50,500),("M",1,10),("K",2,50),("prob_rewire",0.001,0.5), ("set_seed",0,10000), ("culture_momentum",0,20),("learning_error_scale",0,0.5),("alpha_attract",0.1,10),("beta_attract",0.1,10),("alpha_threshold",0.1,10),("beta_threshold",0.1,10)]
    D_vars = len(variable_parameters)
    samples = N_samples * (D_vars + 2)
    #print("samples",samples)
    names_list = [x[0] for x in variable_parameters]
    bounds_list = [[x[1],x[2]] for x in variable_parameters]
    #print(D_vars,names_list,bounds_list)
    problem = {
        "num_vars": D_vars,
        "names": names_list,
        "bounds": bounds_list,
    }

    # Fixed params
    save_data = False
    opinion_dynamics = "DEGROOT"  # "SELECT"
    #K = 20  # k nearest neighbours INTEGER
    #M = 3  # number of behaviours
    #N = 100  # number of agents
    set_seed = 1  ##reproducibility INTEGER
    np.random.seed(set_seed)
    #phi_list = #np.linspace(0.9, 1, num=M)
    #carbon_emissions = []#np.linspace(0.5, 1, num=M)
    
    #np.random.shuffle(carbon_emissions)
    #alpha_attract = 2  ##inital distribution parameters - doing the inverse inverts it!
    #beta_attract = 8
    #alpha_threshold = 8
    #beta_threshold = 2
    total_time = 10
    delta_t = 0.1  # time step size
    time_steps_max = int(
        total_time / delta_t
    )  # number of time steps max, will stop if culture converges
    #prob_rewire = 0.1
    #culture_momentum = 5
    #learning_error_scale = 0.05

    fixed_params = [
        opinion_dynamics,
        save_data,
        time_steps_max,
        delta_t
    ]
            #M,
        #N,
        #phi_list,
        #carbon_emissions,

    #fileName = produceName_SA(variable_parameters)
    fileName = "results/SA_test_lots"
    createFolderSA(fileName)

    time_per_step = 0.02034465771507133
    print("Predicted run time = ", time_per_step * time_steps_max * samples)

    # GENERATE PARAMETER VALUES
    param_values = saltelli.sample(
        problem, N_samples
    )  # NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters
    #print("param_values = ",param_values)
    Y = np.zeros([param_values.shape[0]])

    # run the thing
    start_time = time.time()
    print("start_time =", time.ctime(time.time()))
    for i, X in enumerate(param_values):
        social_network = generate_data(list(fixed_params) + list(X))
        Y[i] = social_network.total_carbon_emissions/(social_network.N*social_network.M)
    # print(Y)
    time_taken = time.time() - start_time
    print(
        "RUN time taken: %s minutes" % ((time_taken) / 60), "or %s s" % ((time_taken))
    )
    print("Actual time per step= ", time_taken / (time_steps_max * samples))

    ## ANALYZE RESULTS
    Si = sobol.analyze(
        problem, Y
    )  # Si is a Python dict with the keys "S1", "S2", "ST", "S1_conf", "S2_conf", and "ST_conf". The _conf keys store the corresponding confidence intervals, typically with a confidence level of 95%.


    
    
    # Visualize
    dpi_save = 1200
    total, first, second = Si.to_df()
    
    #print(total)
    fig, ax = plt.subplots()
    barplot(total, ax = ax)
    plotName = fileName + "/Plots"
    f = plotName + "/SA.png"
    fig.savefig(f, dpi=dpi_save)
    # Si.plot()

    plt.show()
