import numpy as np
import networkx as nx
from individual import Individual
#from scipy.stats import truncnorm

class Network():

    """ 
        Class for network which holds network
        Properties: Culture, Behaviours
    """

    def __init__(self, N, K, prob_wire, delta_t, M, set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix ,psi,carbon_price_policy_start,carbon_price_init,carbon_price_gradient,carbon_emissions,alpha_attract, beta_attract, alpha_threshold, beta_threshold,phi_list,learning_error_scale):

        self.N = N
        self.K = K
        self.prob_wire = prob_wire
        self.M = M
        self.t = 0
        self.alpha_attract, self.beta_attract, self.alpha_threshold, self.beta_threshold = alpha_attract, beta_attract, alpha_threshold, beta_threshold
    
        self.delta_t = delta_t

        self.set_seed = set_seed
        np.random.seed(self.set_seed)

        #carbon price
        self.carbon_price_policy_start = carbon_price_policy_start
        self.carbon_price_init = carbon_price_init
        self.carbon_price = 0
        self.carbon_price_gradient = carbon_price_gradient
        self.carbon_emissions = carbon_emissions

        #cultural definition
        self.culture_momentum = culture_momentum
        self.culture_div = culture_div

        #infromation provision
        self.attract_information_provision_list = attract_information_provision_list
        self.nu = nu
        self.eta = eta
        self.t_IP_matrix  = t_IP_matrix 
        self.t_IP_list = np.empty(self.M)

        #social learning
        self.phi_list =  phi_list 
        self.learning_error_scale = learning_error_scale  

        #indivdiual learning
        self.psi = psi

        self.weighting_matrix, self.network = self.create_weighting_matrix()  
        #print("self.weighting_matrix",self.weighting_matrix,self.weighting_matrix.size)

        self.init_data_behaviours = self.generate_init_data_behaviours()
        self.agent_list = self.create_agent_list() 
        self.behavioural_attract_matrix = self.calc_behavioural_attract_matrix()
        self.social_component_matrix = self.calc_social_component_matrix()

        #self.attract_cultural_group = self.calc_attract_cultural_group()

        self.total_carbon_emissions = self.calc_total_emissions()
        self.average_culture,self.cultural_var,self.min_culture,self.max_culture = self.calc_culture()


        self.update_information_provision()
        self.weighting_matrix_convergence = np.nan#there is no convergence in the first step, to deal with time issues when plotting

        self.history_weighting_matrix = [self.weighting_matrix]
        self.history_behavioural_attract_matrix = [self.behavioural_attract_matrix]
        self.history_social_component_matrix = [self.social_component_matrix]
        self.history_cultural_var = [self.cultural_var]
        self.history_time = [self.t]
        self.history_carbon_price = [self.carbon_price]
        self.history_total_carbon_emissions = [self.total_carbon_emissions]
        self.history_weighting_matrix_convergence = [self.weighting_matrix_convergence]
        self.history_average_culture = [self.average_culture]
        self.history_min_culture = [self.min_culture]
        self.history_max_culture = [self.max_culture]

    def create_weighting_matrix(self):# here is where i need to create a small world transmission matrix
        #SMALL WORLD
        ws = nx.watts_strogatz_graph(n=self.N, k=self.K, p=self.prob_wire, seed=self.set_seed)# Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])

        weighting_matrix = nx.to_numpy_array(ws)

        actual_connections = weighting_matrix.sum()
        #print("actual_connections",actual_connections)
        potential_connections = (self.N*(self.N-1))/2
        #print("potential_connections",potential_connections)
        network_density = actual_connections/potential_connections
        print("network_density = ",network_density)

        #I got this off the internet, I dont really understand how np.newaxis works
        row_sums =  weighting_matrix.sum(axis=1)
        weighting_matrix = weighting_matrix/ row_sums[:, np.newaxis]

        return weighting_matrix, ws 
    
    def generate_init_data_behaviours(self):
        #init_value, init_attract, init_threshold,behaviour_cap, carbon_emissions,attract_individual_learning
        init_data_behaviours = []            
        attract_individual_learning = [1]*self.M #np.random.random_sample(self.M)

        for i in range(self.N):
            row_init_data_behaviours = []
            #### HERE I WHERE I DEFINE THE STARTING VALUES!!

            attract_list = np.random.beta(self.alpha_attract, self.beta_attract, size= self.M)#np.random.random_sample(self.M) #(np.random.random_sample(self.M) - 0.5)*2
            threshold_list = np.random.beta(self.alpha_threshold, self.beta_threshold, size= self.M)#np.random.random_sample(self.M) #these are random samples between 0 and 1
            #init_value, init_attract, init_threshold,behaviour_cap, carbon_emissions,attract_individual_learning,psi
            for v in range(self.M):
                row_init_data_behaviours.append([attract_list[v], threshold_list[v], self.carbon_emissions[v], attract_individual_learning[v], self.psi, self.phi_list[v],self.learning_error_scale])
            init_data_behaviours.append(row_init_data_behaviours)

        return init_data_behaviours

    def create_agent_list(self):
        agent_list = []
        for i in range(self.N):
            agent_list.append(Individual(self.init_data_behaviours[i],self.delta_t,self.culture_momentum,self.attract_information_provision_list,self.nu,self.eta,self.t,self.t_IP_list))#init_data_behaviours, delta_t

        return agent_list

    def calc_behavioural_attract_matrix(self):
        behavioural_attract_matrix = []
        #print("HEY1",self.N,self.M,np.asarray(behavioural_attract_matrix).size)
        #print(behavioural_attract_matrix)
        
        for i in range(self.N):
            row_behavioural_attract_matrix = []
            for v in range(self.M):
                row_behavioural_attract_matrix.append(self.agent_list[i].behaviour_list[v].attract)#do i want attraction or behavioural value?
            #print("row",row_behavioural_attract_matrix)
            behavioural_attract_matrix.append(row_behavioural_attract_matrix)
       
        #print("HEY2",self.N,self.M,np.asarray(behavioural_attract_matrix).size)
        #print(behavioural_attract_matrix)
        return behavioural_attract_matrix

    def calc_social_component_matrix(self):
        
        weighting_matrix_array = np.array(self.weighting_matrix)
        #print("weighting_matrix_array",weighting_matrix_array,weighting_matrix_array.size)
        behavioural_attract_matrix_array = np.array(self.behavioural_attract_matrix)
        #print("behavioural_attract_matrix_array",behavioural_attract_matrix_array,behavioural_attract_matrix_array.size)

        N_M_matrix = np.matmul(weighting_matrix_array,behavioural_attract_matrix_array)

        return N_M_matrix 

    """
    def calc_attract_cultural_group(self):
        N_green = 0

        attract_cultural_group = np.zeros(self.M)

        for i in range(self.N):
            if self.agent_list[i].culture > self.culture_div:
                N_green +=1
                for j in range(self.M):
                    attract_cultural_group[j] += self.agent_list[i].behaviour_list[j].attract

        if N_green == 0:
            return attract_cultural_group
        else:
            attract_cultural_group = attract_cultural_group/N_green
            return attract_cultural_group
    """
    
    def update_information_provision(self):
        for i in range(len(self.t_IP_matrix)):
            if(self.t in self.t_IP_matrix[i]):
                self.t_IP_list[i] = self.t
        
        for i in range(self.N):
            self.agent_list[i].t_IP_list = self.t_IP_list
    
    def update_carbon_price(self):
        #print("time, start ,carbon price",self.t,round(self.t,3), self.carbon_price_policy_start,self.carbon_price)
        if round(self.t,3) == self.carbon_price_policy_start:
            self.carbon_price = self.carbon_price_init
        elif self.t > self.carbon_price_policy_start:
            self.carbon_price += self.carbon_price_gradient
        

    def update_weightings(self):
        #step4, equation 8
        copy_weighting_matrix = np.copy(self.weighting_matrix)
        #print("STEP")

        weighting_matrix = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if self.weighting_matrix[i][j] > 0:#no self interaction (included in the min requiremetn)
                    #self.weighting_matrix[i][j] += self.delta_t*(1 - abs(self.agent_list[i].culture - self.agent_list[j].culture ))
                    #print(self.agent_list[i].culture - self.agent_list[j].culture)
                    #print(np.exp(-abs(self.agent_list[i].culture - self.agent_list[j].culture)))
                    #weighting_matrix[i][j] = np.exp(-abs(self.agent_list[i].culture - self.agent_list[j].culture))
                    weighting_matrix[i][j] = 1 - 0.5*abs(self.agent_list[i].culture - self.agent_list[j].culture)
                    #print(self.agent_list[i].culture - self.agent_list[j].culture,weighting_matrix[i][j])
        
        for i in range(self.N):
            i_total = sum(weighting_matrix[i])
            #print(i_total)
            for j in range(self.N):
                    self.weighting_matrix[i][j] = weighting_matrix[i][j]/i_total

        #print(copy_weighting_matrix,self.weighting_matrix)
        difference_matrix = np.subtract(copy_weighting_matrix,self.weighting_matrix)
        #print(difference_matrix)
        return (np.abs(difference_matrix)).sum()

    def calc_total_emissions(self):
        return sum([x.carbon_emissions for x in self.agent_list])
    
    def calc_culture(self):
        culture_list = [x.culture for x in self.agent_list]
        return np.mean(culture_list), max(culture_list) - min(culture_list), max(culture_list), min(culture_list)

    def save_data_network(self):
        self.history_time.append(self.t)
        self.history_weighting_matrix.append(self.weighting_matrix)
        self.history_behavioural_attract_matrix.append(self.behavioural_attract_matrix)
        self.history_social_component_matrix.append(self.social_component_matrix)
        self.history_carbon_price.append(self.carbon_price)
        self.history_total_carbon_emissions.append(self.total_carbon_emissions)
        self.history_weighting_matrix_convergence.append(self.weighting_matrix_convergence)
        self.history_average_culture.append(self.average_culture)
        self.history_cultural_var.append(self.cultural_var)
        self.history_min_culture.append(self.min_culture)
        self.history_max_culture.append(self.max_culture)

    def next_step(self):
        #advance a time step
        self.t += self.delta_t
        self.update_information_provision()
        self.update_carbon_price()
        for i in range(self.N):
            self.agent_list[i].t = self.t
            self.agent_list[i].next_step(self.social_component_matrix[i], self.carbon_price)#, self.attract_cultural_group,
        self.weighting_matrix_convergence = self.update_weightings()
        self.behavioural_attract_matrix = self.calc_behavioural_attract_matrix()
        self.social_component_matrix = self.calc_social_component_matrix()
        #self.attract_cultural_group = self.calc_attract_cultural_group()

        self.average_culture,self.cultural_var,self.min_culture,self.max_culture = self.calc_culture()
        self.total_carbon_emissions = self.calc_total_emissions()
        self.save_data_network()
