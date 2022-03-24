import numpy as np
import networkx as nx
from individual import Individual

class Network():

    """ 
        Class for network which holds network
        Properties: Culture, Behaviours
    """

    def __init__(self, P, K, prob_wire, delta_t, Y, name_list,behave_type_list, behaviour_cap,set_seed):
        #self.network = self.create_network(network_properties)
        self.agent_number = P
        self.K = K
        self.prob_wire = prob_wire
        self.num_behaviours = Y
        self.delta_t = delta_t
        self.name_list = name_list
        self.behaviour_cap = behaviour_cap
        self.behave_type_list = behave_type_list
        self.set_seed = set_seed
        np.random.seed(self.set_seed)

        self.weighting_matrix, self.network = self.create_weighting_matrix()  
        self.init_data_behaviours = self.generate_init_data_behaviours()
        self.agent_list = self.create_agent_list() 
        self.behavioural_attract_matrix = self.create_update_behavioural_attract_matrix()
        self.social_component_matrix = self.calc_social_component_matrix()
        self.cultural_var = self.calc_cultural_var()

        self.history_weighting_matrix = [self.weighting_matrix]
        self.history_behavioural_attract_matrix = [self.behavioural_attract_matrix]
        self.history_social_component_matrix = [self.social_component_matrix]
        self.history_cultural_var = [self.cultural_var]

    def create_weighting_matrix(self):# here is where i need to create a small world transmission matrix
        #SMALL WORLD
        ws = nx.watts_strogatz_graph(n=self.agent_number, k=self.K, p=self.prob_wire, seed=self.set_seed)# Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
        weighting_matrix = nx.to_numpy_array(ws)

        return weighting_matrix, ws 
    
    def generate_init_data_behaviours(self):
        #behaviour_name, behaviour_type, init_value, init_attract, init_cost
        init_data_behaviours = []

        for i in range(self.agent_number):
            row_init_data_behaviours = []
            #### HERE I WHERE I DEFINE THE STARTING VALUES!!

            attract_list = np.random.random_sample(self.num_behaviours) #(np.random.random_sample(self.num_behaviours) - 0.5)*2
            cost_list = np.random.random_sample(self.num_behaviours)

            for v in range(self.num_behaviours):
                row_init_data_behaviours.append([self.name_list[v],self.behave_type_list[v], attract_list[v] - cost_list[v],attract_list[v],cost_list[v], self.behaviour_cap])
            init_data_behaviours.append(row_init_data_behaviours)

        return init_data_behaviours

    def create_agent_list(self):
        agent_list = []
        for i in range(self.agent_number):
            agent_list.append(Individual(self.init_data_behaviours[i],self.delta_t))#init_data_behaviours, delta_t

        return agent_list

    def create_update_behavioural_attract_matrix(self):
        behavioural_attract_matrix = []
        for i in range(self.agent_number):
            row_behavioural_attract_matrix= []
            for v in range(self.num_behaviours):
                row_behavioural_attract_matrix.append(self.agent_list[i].behaviour_list[v].attract)#do i want attraction or behavioural value?
            behavioural_attract_matrix.append(row_behavioural_attract_matrix)
            
        return behavioural_attract_matrix

    def calc_social_component_matrix(self):
        weighting_matrix_array = np.array(self.weighting_matrix)
        behavioural_attract_matrix_array = np.array(self.behavioural_attract_matrix)

        P_Y_matrix = np.matmul(weighting_matrix_array,behavioural_attract_matrix_array)

        return P_Y_matrix 


    def update_weightings(self):
        #step4, equation 8

        for i in range(self.agent_number):
            for j in range(self.agent_number):
                if self.weighting_matrix[i][j] > 0:#no self interaction (included in the min requiremetn)
                    self.weighting_matrix[i][j] += self.delta_t*(1 - abs(self.agent_list[i].culture - self.agent_list[j].culture ))
        
        for i in range(self.agent_number):
            i_total = sum(self.weighting_matrix[i])
            for j in range(self.agent_number):
                    #print('BERFORE',self.weighting_matrix[i][j])
                    self.weighting_matrix[i][j] = self.weighting_matrix[i][j]/i_total
                    #print('AFTER',self.weighting_matrix[i][j])

    def calc_cultural_var(self):
        culture_list = [x.culture for x in self.agent_list]
        self.cultural_var = max(culture_list) - min(culture_list)

    def save_data_network(self):
        self.history_weighting_matrix.append(self.weighting_matrix)
        self.history_behavioural_attract_matrix.append(self.behavioural_attract_matrix)
        self.history_social_component_matrix.append(self.social_component_matrix)
        self.history_cultural_var.append(self.cultural_var)

    def next_step(self):
        #advance a time step

        for i in range(self.agent_number):
            self.agent_list[i].next_step(self.social_component_matrix[i])
        #self.update_weightings() # TURN OFF FOR NOW 
        self.behavioural_attract_matrix = self.create_update_behavioural_attract_matrix()
        self.social_component_matrix = self.calc_social_component_matrix()
        self.calc_cultural_var()
        self.save_data_network()
