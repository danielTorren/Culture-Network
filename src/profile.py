from network_alt import Network
import time

if __name__ == "__main__":
    save_data = False
    opinion_dynamics = "DEGROOT"  # "SELECT"
    K = 10  # k nearest neighbours INTEGER
    M = 5  # number of behaviours
    N = 100  # number of agents
    total_time = 10
    alpha_attract = 8  ##inital distribution parameters - doing the inverse inverts it!
    beta_attract = 2
    alpha_threshold = 2
    beta_threshold = 8
    delta_t = 0.1  # time step size
    time_steps_max = int(
        total_time / delta_t
    )
    culture_momentum = 5
    set_seed = 1  ##reproducibility INTEGER
    phi_list_lower,phi_list_upper = 0.8,1
    prob_rewire = 0.1  # re-wiring probability?
    culture_momentum = 1  # real time over which culture is calculated for INTEGER
    learning_error_scale = 0.01  # 1 standard distribution is 2% error


    #CURRENT RECORD: 0.3097529411315918
    #DO NOT TOUCH PARAMETER VALUES

    params = {
        "opinion_dynamics": opinion_dynamics,
        "save_data": save_data, 
        "time_steps_max": time_steps_max, 
        "delta_t": delta_t,
        "phi_list_lower": phi_list_lower,
        "phi_list_upper": phi_list_upper,
        "N": N,
        "M": M,
        "K": K,
        "prob_rewire": prob_rewire,
        "set_seed": set_seed,
        "culture_momentum": culture_momentum,
        "learning_error_scale": learning_error_scale,
        "alpha_attract": alpha_attract,
        "beta_attract": beta_attract,
        "alpha_threshold": alpha_threshold,
        "beta_threshold": beta_threshold
    }
    #start_time = time.time()
    social_network = Network(params)
    
    #### RUN TIME STEPS
    time_counter = 0
    while time_counter < params["time_steps_max"]:
        social_network.next_step()
        time_counter += 1
    
    #print(
    #    "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
    #    "or %s s" % ((time.time() - start_time)),
    #)

 