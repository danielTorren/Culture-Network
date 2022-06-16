import numpy as np
import networkx as nx
from individual import Individual

class Network():

    """ 
        Class for network which holds network
        Properties: Culture, Behaviours
    """

    def __init__(self,M,N,phi_list,carbon_emissions,alpha_attract,beta_attract,alpha_threshold,beta_threshold,delta_t,    K,prob_rewire,set_seed,culture_momentum_time,learning_error_scale):

        self.N = N
        self.K = int(round(K))#round due to the sampling method producing floats, lets hope this works
        self.prob_wire = prob_rewire
        self.M = M
        self.t = 0
        self.alpha_attract, self.beta_attract, self.alpha_threshold, self.beta_threshold = alpha_attract, beta_attract, alpha_threshold, beta_threshold
        self.carbon_emissions = carbon_emissions
        self.delta_t = delta_t

        self.set_seed = int(round(set_seed))
        np.random.seed(self.set_seed)#not sure if i have to be worried about the randomness of the system being reproducible


        #cultural definition
        self.culture_momentum = int(round(culture_momentum_time)/self.delta_t)#round due to the sampling method producing floats, lets hope this works


        #social learning
        self.phi_list =  phi_list 
        self.learning_error_scale = learning_error_scale  

        #create network
        self.weighting_matrix, self.network,  self.ego_networks = self.create_weighting_matrix() 

        #create indviduals
        self.init_data_behaviours = self.generate_init_data_behaviours()
        self.agent_list = self.create_agent_list() 

        self.behavioural_attract_matrix = self.calc_behavioural_attract_matrix()
        self.social_component_matrix = self.calc_social_component_matrix()

        self.total_carbon_emissions = self.calc_total_emissions()
        self.cultural_var = self.calc_culture_var()


    def normlize_matrix(self,matrix):
        row_sums =  matrix.sum(axis=1)
        norm_matrix = matrix/ row_sums[:, np.newaxis]

        return norm_matrix

    def create_weighting_matrix(self):# here is where i need to create a small world transmission matrix
        #SMALL WORLD
        ws = nx.watts_strogatz_graph(n=self.N, k=self.K, p=self.prob_wire, seed=self.set_seed)# Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
        weighting_matrix = nx.to_numpy_array(ws)
        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)
        ego_networks = [[n for n in ws[x]] for x in range(self.N)]#create list of neighbours of individuals by index

        return norm_weighting_matrix, ws ,ego_networks #num_neighbours, 

    def generate_init_data_behaviours(self):
        ###init_attract, init_threshold,carbon_emissions
        attract_matrix = [np.random.beta(self.alpha_attract, self.beta_attract, size= self.M) for i in range(self.N)]#np.random.random_sample(self.M) #(np.random.random_sample(self.M) - 0.5)*2
        threshold_matrix = [np.random.beta(self.alpha_threshold, self.beta_threshold, size= self.M) for i in range(self.N)]#np.random.random_sample(self.M) #these are random samples between 0 and 1
        init_data_behaviours = [[[attract_matrix[i][v], threshold_matrix[i][v], self.carbon_emissions[v]] for v in range(self.M)] for i in range(self.N)]
        
        return init_data_behaviours

    def create_agent_list(self):
        agent_list = [Individual(self.init_data_behaviours[i],self.delta_t,self.culture_momentum,self.t,self.M) for i in range(self.N)]
        return agent_list

    def calc_behavioural_attract_matrix(self):
        behavioural_attract_matrix = np.array([[i.behaviour_list[v].attract for v in range(self.M)] for i in self.agent_list])
        return behavioural_attract_matrix

    
    def calc_social_component_matrix(self):
        N_M_matrix = np.zeros((self.N,self.M))
        for i in range(self.N):
            k = np.random.choice(self.ego_networks[i],1)[0]
            for j in range(self.M):
                N_M_matrix[i][j] = self.phi_list[j]*self.weighting_matrix[i][k]*self.behavioural_attract_matrix[i][j] + np.random.normal(loc = 0, scale = self.learning_error_scale)
        return N_M_matrix
        
    def calc_alpha(self, i,j):
        return 1 - 0.5*abs(self.agent_list[i].culture - self.agent_list[j].culture)#calc the un-normalized weighting matrix 

    def update_weightings(self):
        weighting_matrix = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if j in self.ego_networks[i]:#no self interaction (included in the min requiremetn)
                    weighting_matrix[i][j] = self.calc_alpha(i,j)

        self.weighting_matrix =  self.normlize_matrix(weighting_matrix)

    def calc_total_emissions(self):
        return sum([x.carbon_emissions for x in self.agent_list])
    
    def calc_culture_var(self):
        culture_list = [x.culture for x in self.agent_list]
        return max(culture_list) - min(culture_list)

    def update_individuals(self):
        for i in range(self.N):
            self.agent_list[i].t = self.t
            self.agent_list[i].next_step(self.social_component_matrix[i])

    def next_step(self):
        #advance a time step
        self.t += self.delta_t
        #execute step
        self.update_individuals()
        #update network parameters for next step
        self.weighting_matrix_convergence = self.update_weightings()
        self.behavioural_attract_matrix = self.calc_behavioural_attract_matrix()
        self.social_component_matrix = self.calc_social_component_matrix()
        self.cultural_var = self.calc_culture_var()
        self.total_carbon_emissions = self.calc_total_emissions()
