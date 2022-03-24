from network import Network
from utility import produceName,saveObjects,saveData,createFolder
import time

def generate_data(P,  K, prob_wire, delta_t, Y, name_list, behave_type_list, behaviour_cap,set_seed,time_steps_max):
    ### CREATE NETWORK
    social_network = Network( P,  K, prob_wire, delta_t, Y, name_list, behave_type_list, behaviour_cap,set_seed)

    #### RUN TIME STEPS
    convergence = False
    time_counter = 0
    while time_counter < time_steps_max and convergence == False:
        social_network.next_step()
        time_counter += 1
        if social_network.cultural_var < culture_var_min:
            convergence = True

    return social_network

def run(P,  K, prob_wire, delta_t, Y, name_list, behave_type_list, behaviour_cap,set_seed,time_steps_max):
    
    ###GENERATE THE DATA TO BE SAVED
    social_network = generate_data(P,  K, prob_wire, delta_t, Y, name_list, behave_type_list, behaviour_cap,set_seed,time_steps_max)
    steps = len(social_network.history_cultural_var) - 1
    ###SAVE RUN DATA
    fileName, runName = produceName(steps, P,  K, prob_wire, delta_t, Y,behaviour_cap,set_seed)
    dataName = createFolder(fileName)
    saveObjects(social_network, dataName)
    saveData(social_network, dataName)
    print("File Path:", fileName)

if __name__ == "__main__":

    P = 10#number of agents
    K = 3 #k nearest neighbours
    prob_wire = 0.1 #re-wiring probability?
    behaviour_cap = 1
    delta_t = 0.1#time step size
    time_steps_max = 20#number of time steps max, will stop if culture converges
    culture_var_min = 0.01#amount of cultural variation
    set_seed = 1#reproducibility

    name_list = ["pro_env_fuel", "anti_env_fuel","pro_env_transport", "anti_env_transport","pro_env_diet", "anti_env_diet"]
    behave_type_list = [1,0,1,0,1,0]
    Y = len(name_list)#number of behaviours

    start_time = time.time()
    print("start_time =", time.ctime(time.time()))
    ###RUN MODEL
    run(P,  K, prob_wire, delta_t, Y, name_list, behave_type_list, behaviour_cap,set_seed,time_steps_max)

    print ("time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))


