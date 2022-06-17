from run import generate_data
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import time
import matplotlib.pyplot as plt
from SALib.plotting.bar import plot as barplot

if __name__ == "__main__":
    N_samples = 64#1024
    D_vars = 4
    samples = N_samples*(D_vars + 2)
    print("number of runs = ", samples)

    #variable parameters
    """
    
    prob_rewire = 0.1 #re-wiring probability?
    set_seed = 2##reproducibility INTEGER
    culture_momentum = 1#real time over which culture is calculated for INTEGER
    learning_error_scale = 0.02#1 standard distribution is 2% error
    """

    problem = {
        'num_vars': D_vars,
        'names': ['prob_rewire','set_seed','culture_momentum','learning_error_scale'],
        'bounds': [
                [0.01, 0.3],#[3, 30],
                [1, 2*samples],#done so i dont use the same run twice? maybe make this a fixed variable
                [1,50],
                [0.01,0.3]
                ]
    }
    
    #Fixed params
    save_data  = False
    K = 10 #k nearest neighbours INTEGER
    M = 3#number of behaviours
    N = 100#number of agents
    phi_list = [1]*M
    carbon_emissions = [1]*M
    alpha_attract =  2##inital distribution parameters - doing the inverse inverts it!
    beta_attract = 8
    alpha_threshold = 8
    beta_threshold = 2
    total_time = 10
    delta_t = 0.1#time step size
    time_steps_max = int(total_time/delta_t)#number of time steps max, will stop if culture converges
    
    fixed_params = [save_data,time_steps_max,M,N,phi_list,carbon_emissions,alpha_attract,beta_attract,alpha_threshold,beta_threshold,delta_t,K]#10 params

    time_per_step = 0.2
    print("Predicted run time = ", time_per_step*time_steps_max*samples)

    #GENERATE PARAMETER VALUES
    param_values = saltelli.sample(problem, N_samples)#NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters
    #print("param_values = ",param_values)
    Y = np.zeros([param_values.shape[0]])
    
    #run the thing 
    start_time = time.time()
    print("start_time =", time.ctime(time.time()))
    for i, X in enumerate(param_values):
        fixed_params.extend(X)#add the two lists together i think X isnt a list which is why the + doesnt work
        social_network = generate_data(fixed_params)
        Y[i] = social_network.total_carbon_emissions
    #print(Y)
    time_taken = time.time()-start_time
    print ("RUN time taken: %s minutes" % ((time_taken )/60), "or %s s"%((time_taken )))
    print("Actual time per step= ", time_taken/(time_steps_max*samples))

    ## ANALYZE RESULTS
    start_time = time.time()
    print("start_time =", time.ctime(time.time()))
    Si = sobol.analyze(problem, Y) # Si is a Python dict with the keys "S1", "S2", "ST", "S1_conf", "S2_conf", and "ST_conf". The _conf keys store the corresponding confidence intervals, typically with a confidence level of 95%.
    print ("ANALYZE time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))
    
    print("Si",Si)

    #Visualize
    total, first, second = Si.to_df()
    barplot(total)
    #Si.plot()
    plt.show()