import numpy as np
import networkx as nx
from individuals import Individual
import numpy.typing as npt
from random import randrange,seed, shuffle
from scipy.stats import gmean
from logging import raiseExceptions


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

        self.harsh_data = parameters["harsh_data"]
        self.heterogenous_cultural_momentum = parameters["heterogenous_cultural_momentum"]

        #time
        self.t = 0
        self.steps = 0
        self.delta_t = parameters["delta_t"]
        self.averaging_method = parameters["averaging_method"]

        #network
        self.M = int(round(parameters["M"]))
        self.N = int(round(parameters["N"]))
        self.K = int(
            round(parameters["K"])
        )  # round due to the sampling method producing floats, lets hope this works
        self.prob_rewire = parameters["prob_rewire"]

        #culture
        self.culture_momentum_real = parameters["culture_momentum_real"]
        self.culture_momentum = int(
            round(self.culture_momentum_real/ self.delta_t)
        )  # round due to the sampling method producing floats, lets hope this works
        
        if self.heterogenous_cultural_momentum:
            self.quick_changers_prop = parameters["quick_changers_prop"]
            self.lagards_prop  = parameters["quick_changers_prop"]#parameters["lagards_prop"] CHANGE BACK JUST FOR GRAPH

            self.culture_momentum_quick_changers_real = self.culture_momentum_real*parameters["ratio_quick_changers"]#setting what ratios are between the quick changers and the normal individuals
            self.culture_momentum_lagards_real = self.culture_momentum_real*parameters["ratio_lagards"]#setting what ratios are between the lagards and the normal individuals

            self.culture_momentum_quick = int(round(self.culture_momentum_quick_changers_real/ self.delta_t)) 
            self.culture_momentum_lagard= int(round(self.culture_momentum_lagards_real/ self.delta_t)) 
            self.culture_momentum_list = self.generate_heterogenous_cultural_momentum()
            #print("Ral",self.culture_momentum_quick_changers_real,self.culture_momentum_lagards_real, self.culture_momentum_real )
            #self.alpha_quick_changers_cultural_momentum = parameters["alpha_quick_changers_cultural_momentum"]
            #self.beta_quick_changers_cultural_momentum = parameters["beta_quick_changers_cultural_momentum"]
            #self.alpha_lagards_cultural_momentum = parameters["alpha_lagards_cultural_momentum"]
            #self.beta_lagards_cultural_momentum = parameters["beta_lagards_cultural_momentum"]
            if self.culture_momentum_quick == 0.0:
                raiseExceptions("Culture momentum of quick changers zero, increase to at least dt")
        else: 
            self.culture_momentum_list = [self.culture_momentum]*self.N
            
        np.random.shuffle(self.culture_momentum_list)

        #print("culture_momentum_list ",self.culture_momentum_list )

        #time discounting 
        self.discount_factor = parameters["discount_factor"]
        self.present_discount_factor = parameters["present_discount_factor"]
        self.normalized_discount_array = self.calc_normalized_discount_array()
            
        #social learning and bias
        self.confirmation_bias = parameters["confirmation_bias"]
        self.learning_error_scale = parameters["learning_error_scale"]

        #social influence of behaviours
        self.phi_array = np.linspace(parameters["phi_list_lower"], parameters["phi_list_upper"], num=self.M)# CONSPICUOUS CONSUMPTION OF BEHAVIOURS - THE HIgher the more social prestige associated with this behaviour
        
        #emissions associated with each behaviour
        self.carbon_emissions = [1]*self.M #parameters["carbon_emissions"], removed for the sake of the SA

        #network homophily
        self.inverse_homophily = parameters["inverse_homophily"]#0-1
        self.homophilly_rate = parameters["homophilly_rate"]
        self.shuffle_reps = int(round((self.N**self.homophilly_rate)*self.inverse_homophily))#int(round((self.N)*self.inverse_homophily))#im going to square it
        
        if self.harsh_data:
            (
                self.green_extreme_max,
                self.green_extreme_min,
                self.green_extreme_prop,
                self.indifferent_max, 
                self.indifferent_min, 
                self.indifferent_prop,
                self.brown_extreme_min, 
                self.brown_extreme_max,
                self.brown_extreme_prop,
            ) = (parameters["green_extreme_max"], parameters["green_extreme_min"], parameters["green_extreme_prop"], parameters["indifferent_max"], parameters["indifferent_min"], parameters["indifferent_prop"], parameters["brown_extreme_min"], parameters["brown_extreme_max"], parameters["brown_extreme_prop"])
        else:
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
        self.attract_matrix_init, self.threshold_matrix_init = self.generate_init_data_behaviours()#self.generate_init_data_behaviours()#self.generate_init_data_behaviours_homo()
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

            self.green_adoption = self.calc_green_adoption()

            self.history_weighting_matrix = [self.weighting_matrix]
            self.history_social_component_matrix = [self.social_component_matrix]
            self.history_cultural_var = [self.cultural_var]
            self.history_time = [self.t]
            self.history_total_carbon_emissions = [self.total_carbon_emissions]
            self.history_weighting_matrix_convergence = [self.weighting_matrix_convergence]
            self.history_average_culture = [self.average_culture]
            self.history_min_culture = [self.min_culture]
            self.history_max_culture = [self.max_culture]
            self.history_green_adoption = [self.green_adoption]

            if self.carbon_price_state:
                self.history_carbon_price = [self.carbon_price]
            if self.information_provision_state:
                self.history_information_provision = []

    def normlize_matrix(self, matrix: npt.NDArray) ->  npt.NDArray:
        row_sums = matrix.sum(axis=1)
        norm_matrix = matrix/row_sums[:, np.newaxis]

        return norm_matrix

    def calc_normalized_discount_array(self):
        normalized_discount_array = []
        for i in self.culture_momentum_list:
            discount_row = []
            for v in range(i):
                 discount_row.append( self.present_discount_factor*(self.discount_factor)**(self.delta_t*v))
            discount_row[0] = 1.0

            #print(" BERFORE discount_row", discount_row)
            normalized_discount_row = (np.asarray(discount_row)/sum(discount_row)).tolist()
            #print(" AFTER discount_row", discount_row)

            normalized_discount_array.append(normalized_discount_row)
        
        return normalized_discount_array

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

        ego_networks = [
            [i for i in ws[n]] for n in range(self.N)
        ]  # create list of neighbours of individuals by index
        neighbours_list = [[i for i in ws.neighbors(n)] for n in range(self.N)]
        return (
            weighting_matrix,
            norm_weighting_matrix,
            ws,
            ego_networks,
            neighbours_list,
        )  # num_neighbours,

    def generate_heterogenous_cultural_momentum(self):
        quick_changers_cultural_momentum_list  = [self.culture_momentum_quick]*int(round(self.N*self.quick_changers_prop))
        lagards_cultural_momentum_list = [self.culture_momentum_lagard]*int(round(self.N*self.lagards_prop))
        print("len",len(quick_changers_cultural_momentum_list),  len(lagards_cultural_momentum_list))
        normal_cultural_momentum_list = [self.culture_momentum]*(self.N - (len(quick_changers_cultural_momentum_list) + len(lagards_cultural_momentum_list)))# whatever is left over
        total_cultural_momentum_list = quick_changers_cultural_momentum_list+ lagards_cultural_momentum_list + normal_cultural_momentum_list
        return total_cultural_momentum_list

    
    def generate_harsh_data(self):
        attract_matrix_green = [np.random.beta(self.green_extreme_max, self.green_extreme_min, size=self.M) for n in range(int(self.N*self.green_extreme_prop))]
        threshold_matrix_green = [np.random.beta(self.green_extreme_min, self.green_extreme_max, size=self.M)for n in range(int(self.N*self.green_extreme_prop))]

        attract_matrix_indifferent = [np.random.beta(self.indifferent_max, self.indifferent_min, size=self.M) for n in range(int(self.N*self.indifferent_prop))]
        threshold_matrix_indifferent = [np.random.beta(self.indifferent_min, self.indifferent_max, size=self.M)for n in range(int(self.N*self.indifferent_prop))]

        attract_matrix_brown = [np.random.beta(self.brown_extreme_min, self.brown_extreme_max, size=self.M) for n in range(int(self.N*self.brown_extreme_prop))]
        threshold_matrix_brown = [np.random.beta(self.brown_extreme_max, self.brown_extreme_min, size=self.M) for n in range(int(self.N*self.brown_extreme_prop))]

        attract_list = attract_matrix_green + attract_matrix_indifferent + attract_matrix_brown 
        threshold_list = threshold_matrix_green + threshold_matrix_indifferent + threshold_matrix_brown 
        return attract_list, threshold_list
    

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
    
    def quick_av_behaviour(self ,attracts, threshold_weighting_array):

        if self.averaging_method == "Arithmetic":
            return np.mean(attracts)
        elif self.averaging_method == "Geometric":
            return gmean(attracts)
        elif self.averaging_method == "Quadratic":
            return np.sqrt(np.mean(attracts**2))
        elif self.averaging_method == "Threshold weighted arithmetic":
            return np.matmul(threshold_weighting_array, attracts)#/(self.M)
        else:
            raiseExceptions("Invalid averaging method choosen try: Arithmetic or Geometric")

    def quick_indiv_calc_culture(self,attracts, threshold_weighting_array, normalized_discount_vector,culture_momentum ) -> float:
        """
        Calc the individual culture of the attraction matrix for homophilly
        """
        av_behaviour = self.quick_av_behaviour(attracts,threshold_weighting_array)
        #av_behaviour = attracts.sum()/attracts.shape
        av_behaviour_list = [av_behaviour]*culture_momentum
        #### HERE I HAVE THE CORRECT LIST OF AV BEAHVIOUR
        indiv_cul = np.matmul(normalized_discount_vector, av_behaviour_list)
        return indiv_cul

    def quick_calc_culture(self,attract_matrix,threshold_matrix):
        """
        Create culture list from the attraction matrix for homophilly
        """

        cul_list = []
        for i in range(len(attract_matrix)):
            threshold_weighting_array = threshold_matrix[i]/sum(threshold_matrix[i])
            
            cul_list.append(self.quick_indiv_calc_culture(attract_matrix[i],threshold_weighting_array,self.normalized_discount_array[i], self.culture_momentum_list[i]))

        return cul_list


    def generate_init_data_behaviours(self) -> tuple:

        if self.harsh_data:
            attract_list, threshold_list = self.generate_harsh_data()
        else:
                attract_list = [np.random.beta(self.alpha_attract, self.beta_attract, size=self.M) for n in range(self.N)]
                threshold_list = [np.random.beta(self.alpha_threshold, self.beta_threshold, size=self.M) for n in range(self.N)]
        
        attract_matrix = np.asarray(attract_list)
        threshold_matrix = np.asarray(threshold_list)
        
        culture_list = self.quick_calc_culture(attract_matrix,threshold_matrix)#,threshold_matrix
        
        #shuffle the indexes!
        attract_list_sorted = [x for _,x in sorted(zip(culture_list,attract_list))]
        attract_array_circular = self.produce_circular_list(attract_list_sorted)
        attract_array_circular_indexes = list(range(len(attract_array_circular)))
        attract_array_circular_indexes_shuffled = self.partial_shuffle(attract_array_circular_indexes, self.shuffle_reps)
        attract_list_sorted_shuffle = [x for _,x in sorted(zip(attract_array_circular_indexes_shuffled,attract_array_circular))]

        return np.asarray(attract_list_sorted_shuffle), threshold_matrix

    def create_agent_list(self) -> list:

        individual_params = {
                "delta_t" : self.delta_t,
                "t" : self.t,
                "M": self.M,
                "save_data" : self.save_data,
                "carbon_emissions" : self.carbon_emissions,
                #"discount_list" : self.discount_list,
                "carbon_price_state" : self.carbon_price_state,
                "information_provision_state" : self.information_provision_state,
                "phi_array": self.phi_array,
                "averaging_method": self.averaging_method,
                "compression_factor": self.compression_factor,
        }

        if self.carbon_price_state:
            individual_params["carbon_price"] = self.carbon_price

        if self.information_provision_state:
            individual_params["attract_information_provision_list"] =  self.attract_information_provision_list
            individual_params["nu"] =  self.nu
            individual_params["eta"] = self.eta
            individual_params["t_IP_list"] = self.t_IP_list


        agent_list = [Individual(individual_params,self.attract_matrix_init[n],self.threshold_matrix_init[n], self.normalized_discount_array[n], self.culture_momentum_list[n]) for n in range(self.N)]

        return agent_list

    def calc_behavioural_attract_matrix(self) ->  npt.NDArray:
        behavioural_attract_matrix = np.array([n.attracts for n in self.agent_list])
        return behavioural_attract_matrix

    def calc_ego_influence_degroot(self) ->  npt.NDArray:
        return np.matmul(self.weighting_matrix, self.behavioural_attract_matrix)

    def calc_ego_influence_alt(self) ->  npt.NDArray:
        k_list = [np.random.choice(range(self.N), 1, p=self.weighting_matrix[n])[0] for n in range(self.N)]#for each indiviudal select a neighbour using the row of the alpha matrix as the probability
        return np.array([self.agent_list[k].attracts for k in k_list])#make a new NxM where each row is what agent n is going to learn from their selected agent k

    def calc_social_component_matrix(self,ego_influence: npt.NDArray) ->  npt.NDArray:
        #return self.phi_array*(ego_influence + np.random.normal(loc=0, scale=self.learning_error_scale, size=(self.N, self.M)) )
        return ego_influence + np.random.normal(loc=0, scale=self.learning_error_scale, size=(self.N, self.M))#NO PHI

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

    def calc_green_adoption(self):
        adoption = 0
        for n in self.agent_list:
            for m in range(self.M):
                if n.values[m] > 0:
                    adoption += 1
        adoption_ratio = adoption/(self.N*self.M)
        return adoption_ratio*100

    def update_individuals(self):
        for i in range(self.N):
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
        self.history_green_adoption.append(self.green_adoption)

        if self.carbon_price_state:
            self.history_carbon_price.append(self.carbon_price)

    def next_step(self):
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
            self.green_adoption = self.calc_green_adoption()            
            if self.steps%self.compression_factor == 0:
                self.save_data_network()
