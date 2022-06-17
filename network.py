import numpy as np
import networkx as nx
from individual import Individual
import time

class Network():

    """ 
        Class for network which holds network
        Properties: Culture, Behaviours
    """

    def __init__(self,parameters):
        #save_data,time_steps_max,M,N,phi_list,carbon_emissions,alpha_attract,beta_attract,alpha_threshold,beta_threshold,delta_t,K,prob_rewire,set_seed,culture_momentum,learning_error_scale
        self.save_data = parameters[0]
        self.M = parameters[2]
        self.N = parameters[3]
        self.phi_list =  parameters[4]
        self.carbon_emissions = parameters[5] 
        self.alpha_attract, self.beta_attract, self.alpha_threshold, self.beta_threshold = parameters[6],parameters[7],parameters[8],parameters[9]
        self.delta_t = parameters[10]
        self.K = int(round(parameters[11]))#round due to the sampling method producing floats, lets hope this works
        self.prob_rewire = parameters[12]
        self.set_seed = int(round(parameters[13]))
        self.culture_momentum = int(round(parameters[14])/self.delta_t)#round due to the sampling method producing floats, lets hope this works
        self.learning_error_scale = parameters[15]  
        
        self.t = 0
        self.list_people = range(self.N)
        np.random.seed(self.set_seed)#not sure if i have to be worried about the randomness of the system being reproducible      

        #create network
        self.weighting_matrix, self.network,  self.ego_networks, self.neighbours_list = self.create_weighting_matrix() 
        #calc_netork density
        #self.calc_network_density()
        #create indviduals
        self.init_data_behaviours = self.generate_init_data_behaviours()
        self.agent_list = self.create_agent_list() 
        self.behavioural_attract_matrix = self.calc_behavioural_attract_matrix()
        self.social_component_matrix = self.calc_social_component_matrix_degroot()
        self.total_carbon_emissions = self.calc_total_emissions()
        self.average_culture,self.cultural_var,self.min_culture,self.max_culture = self.calc_culture()
        self.weighting_matrix_convergence = np.nan#there is no convergence in the first step, to deal with time issues when plotting

        if self.save_data:
            self.history_weighting_matrix = [self.weighting_matrix]
            self.history_behavioural_attract_matrix = [self.behavioural_attract_matrix]
            self.history_social_component_matrix = [self.social_component_matrix]
            self.history_cultural_var = [self.cultural_var]
            self.history_time = [self.t]
            self.history_total_carbon_emissions = [self.total_carbon_emissions]
            self.history_weighting_matrix_convergence = [self.weighting_matrix_convergence]
            self.history_average_culture = [self.average_culture]
            self.history_min_culture = [self.min_culture]
            self.history_max_culture = [self.max_culture]

    def normlize_matrix(self,matrix):
        row_sums =  matrix.sum(axis=1)
        norm_matrix = matrix/ row_sums[:, np.newaxis]

        return norm_matrix

    def calc_network_density(self):
        actual_connections = self.weighting_matrix.sum()
        #print("actual_connections",actual_connections)
        potential_connections = (self.N*(self.N-1))/2
        #print("potential_connections",potential_connections)
        network_density = actual_connections/potential_connections
        print("network_density = ",network_density)

    def create_weighting_matrix(self):# here is where i need to create a small world transmission matrix
        #SMALL WORLD
        ws = nx.watts_strogatz_graph(n=self.N, k=self.K, p=self.prob_rewire, seed=self.set_seed)# Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
        weighting_matrix = nx.to_numpy_array(ws)
        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)
        ego_networks = [[i for i in ws[n]] for n in self.list_people]#create list of neighbours of individuals by index
        neighbours_list = [[i for i in ws.neighbors(n)] for n in self.list_people]
        return norm_weighting_matrix, ws ,ego_networks,neighbours_list #num_neighbours, 

    def generate_init_data_behaviours(self):
        ###init_attract, init_threshold,carbon_emissions
        #print("attract = ",np.random.beta(self.alpha_attract, self.beta_attract, size= self.M) )
        attract_matrix = np.asarray([np.random.beta(self.alpha_attract, self.beta_attract, size= self.M) for n in self.list_people])#np.random.random_sample(self.M) #(np.random.random_sample(self.M) - 0.5)*2
        threshold_matrix = np.asarray([np.random.beta(self.alpha_threshold, self.beta_threshold, size= self.M) for n in self.list_people])#np.random.random_sample(self.M) #these are random samples between 0 and 1
        #print("sahpe",np.shape(attract_matrix))
        #print(self.carbon_emissions)
        init_data_behaviours = [ [ [attract_matrix[n][m], threshold_matrix[n][m], self.carbon_emissions[m],self.phi_list[m]] for m in range(self.M)] for n in self.list_people]
        return init_data_behaviours

    def create_agent_list(self):
        agent_list = [Individual(self.init_data_behaviours[i],self.delta_t,self.culture_momentum,self.t,self.M,self.save_data) for i in self.list_people]
        return agent_list

    def calc_behavioural_attract_matrix(self):
        behavioural_attract_matrix = np.array([[n.behaviour_list[m].attract for m in range(self.M)] for n in self.agent_list])
        return behavioural_attract_matrix

    def calc_social_component_matrix_degroot(self):
        np.random.seed(self.set_seed)
        return np.asarray(self.phi_list)*(np.matmul(self.weighting_matrix,self.behavioural_attract_matrix) + np.random.normal(loc = 0, scale = self.learning_error_scale, size=(self.N, self.M)) - self.behavioural_attract_matrix)# + np.random.normal(loc = 0, scale = self.learning_error_scale, size=(self.N, self.M))
        #np.asarray(self.phi_list)*np.matmul(self.weighting_matrix,self.behavioural_attract_matrix) + np.random.normal(loc = 0, scale = self.learning_error_scale, size=(self.N, self.M))

    def calc_social_component_matrix_alt(self):
        k_list = [np.random.choice(self.list_people,1, p = self.weighting_matrix[n])[0] for n in self.list_people] 
        re_order_list = [self.agent_list[k] for k in k_list]
        behavioural_attract_matrix_k = np.array([[k.behaviour_list[m].attract for m in range(self.M)] for k in re_order_list])

        return np.asarray(self.phi_list)*(behavioural_attract_matrix_k + np.random.normal(loc = 0, scale = self.learning_error_scale, size=(self.N, self.M)) - self.behavioural_attract_matrix)





    def calc_alpha(self, i,j):
        return 1 - 0.5*abs(self.agent_list[i].culture - self.agent_list[j].culture)#calc the un-normalized weighting matrix 

    def calc_total_weighting_matrix_difference(self,matrix_before, matrix_after):
        difference_matrix = np.subtract(matrix_before,matrix_after)
        total_difference = (np.abs(difference_matrix)).sum()
        return total_difference
    
    def update_weightings(self):

        weighting_matrix = np.zeros((self.N,self.N))
        for i in self.list_people:
            for j in self.list_people:
                if j in self.ego_networks[i]:#no self interaction (included in the min requiremetn)
                    weighting_matrix[i][j] = self.calc_alpha(i,j)

        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)
        total_difference = self.calc_total_weighting_matrix_difference(self.weighting_matrix, norm_weighting_matrix)
        
        #set the new weightign matrix
        self.weighting_matrix = norm_weighting_matrix
        
        return total_difference

    def calc_total_emissions(self):
        return sum([x.carbon_emissions for x in self.agent_list])
    
    def calc_culture(self):
        culture_list = [x.culture for x in self.agent_list]
        return np.mean(culture_list), max(culture_list) - min(culture_list), max(culture_list), min(culture_list)

    def update_individuals(self):
        for i in self.list_people:
            self.agent_list[i].t = self.t
            self.agent_list[i].next_step(self.social_component_matrix[i])

    def save_data_network(self):
        self.history_time.append(self.t)
        self.history_weighting_matrix.append(self.weighting_matrix)
        self.history_behavioural_attract_matrix.append(self.behavioural_attract_matrix)
        self.history_social_component_matrix.append(self.social_component_matrix)
        self.history_total_carbon_emissions.append(self.total_carbon_emissions)
        self.history_weighting_matrix_convergence.append(self.weighting_matrix_convergence)
        self.history_average_culture.append(self.average_culture)
        self.history_cultural_var.append(self.cultural_var)
        self.history_min_culture.append(self.min_culture)
        self.history_max_culture.append(self.max_culture)

    def next_step(self):
        #advance a time step
        self.t += self.delta_t

        #execute step
        self.update_individuals()

        #update network parameters for next step
        self.weighting_matrix_convergence = self.update_weightings()
        self.behavioural_attract_matrix = self.calc_behavioural_attract_matrix()

        self.social_component_matrix = self.calc_social_component_matrix_degroot()

        self.average_culture,self.cultural_var,self.min_culture,self.max_culture = self.calc_culture()
        self.total_carbon_emissions = self.calc_total_emissions()
        #print(self.save_data)
        if self.save_data: 
            self.save_data_network()


