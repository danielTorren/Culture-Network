from network import Network

def generate_data(parameters):
    ### CREATE NETWORK
    #                    
    social_network = Network(parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],parameters[8],parameters[9],parameters[10],parameters[11],parameters[12],parameters[13],parameters[14])
    time_steps_max = parameters[0]
    #### RUN TIME STEPS
    time_counter = 0
    while time_counter < time_steps_max:
        social_network.next_step()
        time_counter += 1

    return social_network