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
    N_samples = 2  # 1024
    
    # variable parameters
    variable_parameters = [("time_steps_max", 10,1000),("phi_list_lower",0,0.7),("phi_list_upper",0.7,1),("N",50,500),("M",1,10),("K",2,50),("prob_rewire",0.001,0.5), ("set_seed",0,10000), ("culture_momentum",1,20),("learning_error_scale",0,0.5)]
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
    total_time = 10
    delta_t = 0.01  # time step size
    time_steps_max = int(
        total_time / delta_t
    )  # number of time steps max, will stop if culture converges

    
    params = {
        "opinion_dynamics": "DEGROOT",
        "save_data": False, 
        "delta_t": delta_t,
        "alpha_attract":1,
        "beta_attract":1,
        "alpha_threshold":1,
        "beta_threshold":1,
    }

    def add_varaiables_to_dict(params,variable_parameters,X):
        for i in range(len(X)):
            params[variable_parameters[i][0]] = X[i]
        return params

    #fileName = produceName_SA(variable_parameters)
    fileName = "results/SA_%s_%s_%s" % (str(samples),str(D_vars),str(N_samples))
    print("fileName: ", fileName)
    createFolderSA(fileName)

    time_per_step = 0.015515416612227757
    #print("Predicted run time = ", time_per_step * time_steps_max * samples, "s")

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
        variable_params_added = add_varaiables_to_dict(params,variable_parameters,X)
        social_network = generate_data(variable_params_added)
        Y[i] = social_network.total_carbon_emissions/(social_network.N*social_network.M)
    
    time_taken = time.time() - start_time
    print(
        "RUN time taken: %s minutes" % ((time_taken) / 60), "or %s s" % ((time_taken))
    )

    print("Actual time per step= ", time_taken / (time_steps_max * samples))

    ## ANALYZE RESULTS
    with open(fileName + "/Data_Y.txt", 'w') as f:
        for i in Y:
            f.write(str(i) + '\n')

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
