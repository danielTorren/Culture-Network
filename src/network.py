import numpy as np
import networkx as nx
from individuals import Individual
import numpy.typing as npt
from random import randrange,seed

class Network:

    """
    Class for network which holds network
    Properties: Culture, Behaviours
    """

    def __init__(self, parameters:list):
        #print(parameters)
        
        self.set_seed = int(round(parameters["set_seed"]))
        seed(self.set_seed)
        np.random.seed(
            self.set_seed
        )  # not sure if i have to be worried about the randomness of the system being reproducible
        
        self.alpha_change = parameters["alpha_change"]
        self.opinion_dyanmics = parameters["opinion_dynamics"]
        self.save_data = parameters["save_data"]
        self.compression_factor = parameters["compression_factor"]
        self.linear_alpha_diff_state = parameters["linear_alpha_diff_state"]
        self.homophily_state = parameters["homophily_state"]

        #time
        self.t = 0
        self.steps = 0
        self.delta_t = parameters["delta_t"]

        #culture
        self.culture_momentum_real = parameters["culture_momentum_real"]
        self.culture_momentum = int(
            round(self.culture_momentum_real/ self.delta_t)
        )  # round due to the sampling method producing floats, lets hope this works

        #time discounting 
        self.discount_factor = parameters["discount_factor"]
        self.present_discount_factor = parameters["present_discount_factor"]
        time_list_beahviours =  np.asarray([self.delta_t*x for x in range(self.culture_momentum)])
        self.discount_list = self.present_discount_factor*(self.discount_factor)**(time_list_beahviours)
        self.discount_list[0] = 1

        #network
        self.M = int(round(parameters["M"]))
        self.N = int(round(parameters["N"]))
        self.list_people = range(self.N)
        self.K = int(
            round(parameters["K"])
        )  # round due to the sampling method producing floats, lets hope this works
        (
            self.alpha_attract,
            self.beta_attract,
            self.alpha_threshold,
            self.beta_threshold,
        ) = (parameters["alpha_attract"], parameters["beta_attract"], parameters["alpha_threshold"], parameters["beta_threshold"])
        self.prob_rewire = parameters["prob_rewire"]
        
        
        #social learning and bias
        self.confirmation_bias = parameters["confirmation_bias"]
        self.learning_error_scale = parameters["learning_error_scale"]

        #social influence of behaviours
        self.phi_array = np.linspace(parameters["phi_list_lower"], parameters["phi_list_upper"], num=self.M)
        #emissions associated with each behaviour
        self.carbon_emissions = [1]*self.M #parameters["carbon_emissions"], removed for the sake of the SA

        #network homophily
        self.inverse_homophily = parameters["inverse_homophily"]#0-1
        self.homophilly_rate = parameters["homophilly_rate"]
        self.shuffle_reps = int(round((self.N**self.homophilly_rate)*self.inverse_homophily))#int(round((self.N)*self.inverse_homophily))#im going to square it
        #print("self.inverse_homophily= ", self.inverse_homophily,"self.shuffle_reps = ",self.shuffle_reps)
        (
            self.alpha_attract,
            self.beta_attract,
            self.alpha_threshold,
            self.beta_threshold,
        ) = (parameters["alpha_attract"], parameters["beta_attract"], parameters["alpha_threshold"], parameters["beta_threshold"])


        #carbon price
        self.carbon_price_state = parameters["carbon_price_state"]
        if self.carbon_price_state:
            self.carbon_price_policy_start = parameters["carbon_price_policy_start"]
            self.carbon_price_init = parameters["carbon_price_init"]
            self.carbon_price = 0
            self.carbon_price_gradient = parameters["carbon_price_gradient"]

        #information provision
        self.information_provision_state = parameters["information_provision_state"]
        if self.information_provision_state:
            self.attract_information_provision_list = parameters["attract_information_provision_list"]
            self.nu = parameters["nu"]
            self.eta = parameters["eta"]
            self.t_IP_matrix  = parameters["t_IP_matrix"]
            self.t_IP_list = np.empty(self.M)

        # create indviduals#Do a homophilly
        
        #if self.homophily_state: 
            #self.attract_matrix_init, self.threshold_matrix_init = self.generate_init_data_behaviours_homo_alt()#self.generate_init_data_behaviours_homo()
        self.attract_matrix_init, self.threshold_matrix_init = self.generate_init_data_behaviours()#self.generate_init_data_behaviours_homo()
        #else:
            #self.attract_matrix_init, self.threshold_matrix_init = self.generate_init_data_behaviours()#self.generate_init_data_behaviours_homo()
        
        #self.attract_matrix_init, self.threshold_matrix_init = self.generate_init_data_behaviours_homo_harsh()

        self.agent_list = self.create_agent_list()

        self.behavioural_attract_matrix = self.calc_behavioural_attract_matrix()#need to leave outside as its a thing being saved, why is it being saved???

        # create network
        (
            self.adjacency_matrix,
            self.weighting_matrix,
            self.network,
            self.ego_networks,
            self.neighbours_list,
        ) = self.create_weighting_matrix()

        if self.opinion_dyanmics == "SELECT":
            ego_influence = self.calc_ego_influence_alt()
        elif self.opinion_dyanmics == "DEGROOT":
            ego_influence = self.calc_ego_influence_degroot()
        else:
            raise Exception("Invalid opinion dynamics model")

        self.social_component_matrix = self.calc_social_component_matrix(ego_influence)

        if self.alpha_change == 1 or 0.5:
            self.weighting_matrix,__ = self.update_weightings()  # Should I update the weighting matrix? I feel like it makes sense if its not time dependant.

        self.total_carbon_emissions = self.calc_total_emissions()

        if self.save_data:
            # calc_netork density
            #self.calc_network_density()

            (
                self.average_culture,
                self.cultural_var,
                self.min_culture,
                self.max_culture,
            ) = self.calc_network_culture()
            self.weighting_matrix_convergence = 0 # there is no convergence in the first step, to deal with time issues when plotting

            self.history_weighting_matrix = [self.weighting_matrix]
            self.history_social_component_matrix = [self.social_component_matrix]
            self.history_cultural_var = [self.cultural_var]
            self.history_time = [self.t]
            self.history_total_carbon_emissions = [self.total_carbon_emissions]
            self.history_weighting_matrix_convergence = [self.weighting_matrix_convergence]
            self.history_average_culture = [self.average_culture]
            self.history_min_culture = [self.min_culture]
            self.history_max_culture = [self.max_culture]

            if self.carbon_price_state:
                self.history_carbon_price = [self.carbon_price]
            if self.information_provision_state:
                self.history_information_provision = []

    def normlize_matrix(self, matrix: npt.NDArray) ->  npt.NDArray:
        row_sums = matrix.sum(axis=1)
        norm_matrix = matrix/row_sums[:, np.newaxis]

        return norm_matrix

    def calc_network_density(self):
        actual_connections = self.weighting_matrix.sum()

        potential_connections = (self.N * (self.N - 1)) / 2

        network_density = actual_connections / potential_connections
        print("network_density = ", network_density)

    def create_weighting_matrix(self)-> tuple[ npt.NDArray, nx.Graph, list, list]:  
        # SMALL WORLD
        ws = nx.watts_strogatz_graph(
            n=self.N, k=self.K, p=self.prob_rewire, seed=self.set_seed
        )  # Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
        weighting_matrix = nx.to_numpy_array(ws)
        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)
        #print("init NOT CULTURED norm_weighting_matrix:",norm_weighting_matrix)

        ego_networks = [
            [i for i in ws[n]] for n in self.list_people
        ]  # create list of neighbours of individuals by index
        neighbours_list = [[i for i in ws.neighbors(n)] for n in self.list_people]
        return (
            weighting_matrix,
            norm_weighting_matrix,
            ws,
            ego_networks,
            neighbours_list,
        )  # num_neighbours,

    """
    def generate_init_data_behaviours_homo_harsh(self) -> tuple:
        ###init_attract, init_threshold,carbon_emissions
        attract_matrix_green = [np.random.beta(8, 2, size=self.M) for n in range(int(self.N/5))]
        threshold_matrix_green = [np.random.beta(2, 8, size=self.M)for n in range(int(self.N/5))]

        attract_matrix_indifferent = [np.random.beta(2, 2, size=self.M) for n in range(int(self.N*3/5))]
        threshold_matrix_indifferent = [np.random.beta(2, 2, size=self.M)for n in range(int(self.N*3/5))]

        attract_matrix_brown = [np.random.beta(2, 8, size=self.M) for n in range(int(self.N/5))]
        threshold_matrix_brown = [np.random.beta(8, 2, size=self.M)for n in range(int(self.N/5))]

        attract_list = attract_matrix_green + attract_matrix_indifferent + attract_matrix_brown 
        threshold_list = threshold_matrix_green + threshold_matrix_indifferent + threshold_matrix_brown 

        attract_matrix = np.asarray(attract_list)
        threshold_matrix = np.asarray(threshold_list)

        culture_list = self.quick_calc_culture(attract_matrix,threshold_matrix)

        #shuffle the indexes!
        attract_list_sorted = [x for _,x in sorted(zip(culture_list,attract_list))]
        threshold_list_sorted = [x for _,x in sorted(zip(culture_list,threshold_list))]

        attract_array_circular = self.produce_circular_list(attract_list_sorted)
        threshold_array_circular = self.produce_circular_list(threshold_list_sorted)

        attract_array_circular_indexes = list(range(len(attract_array_circular)))
        attract_array_circular_indexes_shuffled = self.partial_shuffle(attract_array_circular_indexes, self.shuffle_reps)

        attract_list_sorted_shuffle = [x for _,x in sorted(zip(attract_array_circular_indexes_shuffled,attract_array_circular))]
        threshold_list_sorted_shuffle = [x for _,x in sorted(zip(attract_array_circular_indexes_shuffled,threshold_array_circular))]
        

        return np.asarray(attract_list_sorted_shuffle),np.asarray(threshold_list_sorted_shuffle)
    """

    def produce_circular_list(self,list):
        first_half = list[::2]
        second_half =  (list[1::2])[::-1]
        circular = first_half + second_half
        return circular

    def partial_shuffle(self, l, swap_reps=5):
        n = len(l)
        for _ in range(swap_reps):
            a, b = randrange(n), randrange(n)
            l[b], l[a] = l[a], l[b]
        return l

    def quick_indiv_calc_culture(self,attracts) -> float:
        """
        Calc the individual culture of the attraction matrix for homophilly
        """
        av_behaviour = attracts.sum()/attracts.shape
        av_behaviour_list = [av_behaviour]*self.culture_momentum
        #### HERE I HAVE THE CORRECT LIST OF AV BEAHVIOUR
        indiv_cul = np.matmul(self.discount_list, av_behaviour_list)/sum(self.discount_list)
        return indiv_cul

    def quick_calc_culture(self,attract_matrix):
        """
        Create culture list from the attraction matrix for homophilly
        """

        cul_list = []
        for i in range(len(attract_matrix)):
            cul_list.append(self.quick_indiv_calc_culture(attract_matrix[i]))

        #print("cul_list = ", np.mean(cul_list))
        return cul_list


    def generate_init_data_behaviours(self) -> tuple:
        ###init_attract, init_threshold,carbon_emissions
        attract_list = [np.random.beta(self.alpha_attract, self.beta_attract, size=self.M) for n in self.list_people]
        threshold_list = [np.random.beta(self.alpha_threshold, self.beta_threshold, size=self.M) for n in self.list_people]
        
        attract_matrix = np.asarray(attract_list)
        threshold_matrix = np.asarray(threshold_list)

        culture_list = self.quick_calc_culture(attract_matrix)#,threshold_matrix
        
        #shuffle the indexes!
        attract_list_sorted = [x for _,x in sorted(zip(culture_list,attract_list))]
        #threshold_list_sorted = [x for _,x in sorted(zip(culture_list,threshold_list))]

        attract_array_circular = self.produce_circular_list(attract_list_sorted)
        #threshold_array_circular = self.produce_circular_list(threshold_list_sorted)

        attract_array_circular_indexes = list(range(len(attract_array_circular)))
        attract_array_circular_indexes_shuffled = self.partial_shuffle(attract_array_circular_indexes, self.shuffle_reps)

        attract_list_sorted_shuffle = [x for _,x in sorted(zip(attract_array_circular_indexes_shuffled,attract_array_circular))]
        #threshold_list_sorted_shuffle = [x for _,x in sorted(zip(attract_array_circular_indexes_shuffled,threshold_array_circular))]
        
        #print("culture_list WUICK",self.quick_calc_culture(np.asarray(attract_list_sorted_shuffle)))
        #print("culture attract shuffel ",np.asarray(attract_list_sorted_shuffle))
        return np.asarray(attract_list_sorted_shuffle), threshold_matrix

    def create_agent_list(self) -> list:

        individual_params = {
                "delta_t" : self.delta_t,
                "culture_momentum" : self.culture_momentum,
                "t" : self.t,
                "M": self.M,
                "save_data" : self.save_data,
                "carbon_emissions" : self.carbon_emissions,
                "discount_list" : self.discount_list,
                "carbon_price_state" : self.carbon_price_state,
                "information_provision_state" : self.information_provision_state,
                "compression_factor": self.compression_factor,
        }

        if self.carbon_price_state:
            individual_params["carbon_price"] = self.carbon_price

        if self.information_provision_state:
            individual_params["attract_information_provision_list"] =  self.attract_information_provision_list
            individual_params["nu"] =  self.nu
            individual_params["eta"] = self.eta
            individual_params["t_IP_list"] = self.t_IP_list


        agent_list = [Individual(individual_params,self.attract_matrix_init[n],self.threshold_matrix_init[n]) for n in self.list_people]

        return agent_list

    def calc_behavioural_attract_matrix(self) ->  npt.NDArray:
        behavioural_attract_matrix = np.array([n.attracts for n in self.agent_list])
        return behavioural_attract_matrix

    def calc_ego_influence_degroot(self) ->  npt.NDArray:
        return np.matmul(self.weighting_matrix, self.behavioural_attract_matrix)

    def calc_ego_influence_alt(self) ->  npt.NDArray:
        k_list = [np.random.choice(self.list_people, 1, p=self.weighting_matrix[n])[0] for n in self.list_people]#for each indiviudal select a neighbour using the row of the alpha matrix as the probability
        return np.array([self.agent_list[k].attracts for k in k_list])#make a new NxM where each row is what agent n is going to learn from their selected agent k

    def calc_social_component_matrix(self,ego_influence: npt.NDArray) ->  npt.NDArray:
        return self.phi_array*(ego_influence + np.random.normal(loc=0, scale=self.learning_error_scale, size=(self.N, self.M)) - self.behavioural_attract_matrix)

    def calc_total_weighting_matrix_difference(self, matrix_before: npt.NDArray, matrix_after: npt.NDArray)-> float:
        difference_matrix = np.subtract(matrix_before, matrix_after)
        total_difference = (np.abs(difference_matrix)).sum()
        return total_difference

    def update_information_provision(self):
        for i in range(len(self.t_IP_matrix)):
            if( round(self.t,3) in self.t_IP_matrix[i]):#BODGE!
                self.t_IP_list[i] = self.t
        
        for i in range(self.N):
            self.agent_list[i].t_IP_list = self.t_IP_list

    def update_carbon_price(self):
        if round(self.t,3) == self.carbon_price_policy_start:#BODGE!
            self.carbon_price = self.carbon_price_init
        elif self.t > self.carbon_price_policy_start:
            self.carbon_price += self.carbon_price_gradient

    def cultural_difference_factor_linear(self,difference_matrix):
        return (1 - 0.5*self.confirmation_bias*np.abs(difference_matrix))

    def cultural_difference_factor_exponential(self,difference_matrix):
        return np.exp(-self.confirmation_bias*np.abs(difference_matrix))

    def update_weightings(self)-> float:
        culture_list = np.array([x.culture for x in self.agent_list])

        difference_matrix = np.subtract.outer(culture_list, culture_list)
        if self.linear_alpha_diff_state:
            #linear
            alpha = self.cultural_difference_factor_linear(difference_matrix)
        else:
            #exponential
            alpha = self.cultural_difference_factor_exponential(difference_matrix)

        diagonal = self.adjacency_matrix*alpha
        
        norm_weighting_matrix = self.normlize_matrix(diagonal)

        
        if self.save_data:
            total_difference = self.calc_total_weighting_matrix_difference(
                self.weighting_matrix, norm_weighting_matrix
            )
            return norm_weighting_matrix,total_difference
        else:
            return norm_weighting_matrix,0#BODGE! bodge for mypy

    def calc_total_emissions(self) -> int:
        return sum([x.total_carbon_emissions for x in self.agent_list])

    def calc_network_culture(self) ->  tuple[float, float, float, float]:
        culture_list = [x.culture for x in self.agent_list]
        #print(np.mean(culture_list))
        return (
            np.mean(culture_list),
            max(culture_list) - min(culture_list),
            max(culture_list),
            min(culture_list),
        )

    def update_individuals(self):
        for i in self.list_people:
            if self.carbon_price_state:
                self.agent_list[i].carbon_price = self.carbon_price
            self.agent_list[i].next_step(self.t,self.steps,self.social_component_matrix[i])

    def save_data_network(self):
        self.history_time.append(self.t)
        self.history_weighting_matrix.append(self.weighting_matrix)
        self.history_social_component_matrix.append(self.social_component_matrix)
        self.history_total_carbon_emissions.append(self.total_carbon_emissions)
        self.history_weighting_matrix_convergence.append(
            self.weighting_matrix_convergence
        )
        self.history_average_culture.append(self.average_culture)
        self.history_cultural_var.append(self.cultural_var)
        self.history_min_culture.append(self.min_culture)
        self.history_max_culture.append(self.max_culture)

        if self.carbon_price_state:
            self.history_carbon_price.append(self.carbon_price)

    def next_step(self):
        #print("HEYE")
        # advance a time step
        self.t += self.delta_t
        self.steps += 1

        #unsure where this step should go SORT THIS OUT
        
        if self.information_provision_state:
            self.update_information_provision()
        if self.carbon_price_state:
            self.update_carbon_price()

        # execute step
        self.update_individuals()

        # update network parameters for next step
        if self.alpha_change == 1:
                if self.save_data:
                        self.weighting_matrix,self.weighting_matrix_convergence = self.update_weightings()
                else:
                        self.weighting_matrix,__ = self.update_weightings()
        
        self.behavioural_attract_matrix = self.calc_behavioural_attract_matrix()

        if self.opinion_dyanmics == "SELECT":
            ego_influence = self.calc_ego_influence_alt()
        elif self.opinion_dyanmics == "DEGROOT":
            ego_influence = self.calc_ego_influence_degroot()
        else:
            raise Exception("Invalid opinion dynamics model")

        self.social_component_matrix = self.calc_social_component_matrix(ego_influence)

        self.total_carbon_emissions = self.calc_total_emissions()

        # print(self.save_data)
        if self.save_data:
            (
                self.average_culture,
                self.cultural_var,
                self.min_culture,
                self.max_culture,
            ) = self.calc_network_culture()
            if self.steps%self.compression_factor == 0:
                self.save_data_network()