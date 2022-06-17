from network import Network
from utility import produceName_alt,createFolder,saveObjects,saveData
import time
def generate_data(parameters):
    ### CREATE NETWORK
    ##params = [save_data,time_steps_max,M,N,phi_list,carbon_emissions,alpha_attract,beta_attract,alpha_threshold,beta_threshold,delta_t,K,prob_rewire,set_seed,culture_momentum,learning_error_scale]
    start_time = time.time()
    #print("start_time =", time.ctime(time.time()))
    social_network = Network(parameters)
    time_steps_max = parameters[1]
    #### RUN TIME STEPS
    time_counter = 0
    while time_counter < time_steps_max:
        social_network.next_step()
        time_counter += 1
    #print("why does it go wrong tiem wise: ",len(social_network.history_time), len(social_network.agent_list[0].history_carbon_emissions), len(social_network.agent_list[0].behaviour_list[0].history_value))
    #print("the thing i am plotting",social_network.history_time,social_network.agent_list[0].behaviour_list[0].history_attract)
    print ("SIMULATION time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))
    return social_network

def run(parameters, to_save_list,params_name):

    #params = [save_data,time_steps_max,M,N,phi_list,carbon_emissions,alpha_attract,beta_attract,alpha_threshold,beta_threshold,delta_t,K,prob_rewire,set_seed,culture_momentum,learning_error_scale]

    ###GENERATE THE DATA TO BE SAVED

    social_network = generate_data(parameters)

    #print(social_network.history_cultural_var)
    steps = len(social_network.history_cultural_var)
    M = parameters[2]
    N = parameters[3]
    #print("run steps",steps,M,N)
    start_time = time.time()
    #print("start_time =", time.ctime(time.time()))
    fileName = produceName_alt(params_name)
    #print("fileName",fileName)
    dataName = createFolder(fileName)
    saveObjects(social_network, dataName)
    saveData(social_network, dataName,steps, N, M,to_save_list[0], to_save_list[1],to_save_list[2],to_save_list[3])
    print("File Path:", fileName)
    print ("SAVE time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))
    
    return fileName