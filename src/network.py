import numpy as np
import networkx as nx
from individuals import Individual
import numpy.typing as npt
from logging import raiseExceptions

class Network:

    """
    Class for network which holds network
    Properties: Culture, Behaviours
    """

    def __init__(self, parameters:list):
        """
        # randomly initialize the RNG from some platform-dependent source of entropy
        np.random.seed(None)
        # get the initial state of the RNG
        st0 = np.random.get_state()
        print("seed",st0)
        """
        
        self.set_seed = parameters["set_seed"]
        np.random.seed(self.set_seed)  # not sure if i have to be worried about the randomness of the system being reproducible
        
        self.alpha_change = parameters["alpha_change"]
        self.save_data = parameters["save_data"]
        self.compression_factor = parameters["compression_factor"]
        self.harsh_data = parameters["harsh_data"]

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
        
        self.culture_momentum_list = [self.culture_momentum]*self.N

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
                self.alpha_attitude,
                self.beta_attitude,
                self.alpha_threshold,
                self.beta_threshold,
            ) = (parameters["alpha_attitude"], parameters["beta_attitude"], parameters["alpha_threshold"], parameters["beta_threshold"])

        self.attitude_matrix_init, self.threshold_matrix_init = self.generate_init_data_behaviours()
        self.agent_list = self.create_agent_list()
        self.behavioural_attitude_matrix = self.calc_behavioural_attitude_matrix()

        # create network
        (
            self.adjacency_matrix,
            self.weighting_matrix,
            self.network,
            self.ego_networks,
            self.neighbours_list,
        ) = self.create_weighting_matrix()


        ego_influence = self.calc_ego_influence_degroot()

        self.social_component_matrix = self.calc_social_component_matrix(ego_influence)

        if self.alpha_change != 0.0:
            self.weighting_matrix,__ = self.update_weightings() 


        if self.save_data:
            self.total_carbon_emissions = self.calc_total_emissions()
            
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

            normalized_discount_row = (np.asarray(discount_row)/sum(discount_row)).tolist()

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
    
    def generate_harsh_data(self):
        attitude_matrix_green = [np.random.beta(self.green_extreme_max, self.green_extreme_min, size=self.M) for n in range(int(self.N*self.green_extreme_prop))]
        threshold_matrix_green = [np.random.beta(self.green_extreme_min, self.green_extreme_max, size=self.M)for n in range(int(self.N*self.green_extreme_prop))]

        attitude_matrix_indifferent = [np.random.beta(self.indifferent_max, self.indifferent_min, size=self.M) for n in range(int(self.N*self.indifferent_prop))]
        threshold_matrix_indifferent = [np.random.beta(self.indifferent_min, self.indifferent_max, size=self.M)for n in range(int(self.N*self.indifferent_prop))]

        attitude_matrix_brown = [np.random.beta(self.brown_extreme_min, self.brown_extreme_max, size=self.M) for n in range(int(self.N*self.brown_extreme_prop))]
        threshold_matrix_brown = [np.random.beta(self.brown_extreme_max, self.brown_extreme_min, size=self.M) for n in range(int(self.N*self.brown_extreme_prop))]

        attitude_list = attitude_matrix_green + attitude_matrix_indifferent + attitude_matrix_brown 
        threshold_list = threshold_matrix_green + threshold_matrix_indifferent + threshold_matrix_brown 
        return attitude_list, threshold_list

    def produce_circular_list(self,list):
        first_half = list[::2]
        second_half =  (list[1::2])[::-1]
        circular = first_half + second_half
        return circular

    def partial_shuffle(self, l, swap_reps=5):
        n = len(l)
        for _ in range(swap_reps):
            a, b = np.random.randint(low = 0, high = n, size=2)
            l[b], l[a] = l[a], l[b]
        return l
    
    def quick_av_behaviour(self ,attitudes, threshold_weighting_array):

        if self.averaging_method == "Arithmetic":
            return np.mean(attitudes)
        elif self.averaging_method == "Threshold weighted arithmetic":
            return np.matmul(threshold_weighting_array, attitudes)#/(self.M)
        else:
            raiseExceptions("Invalid averaging method choosen try: Arithmetic or Geometric")

    def quick_indiv_calc_culture(self,attitudes, threshold_weighting_array, normalized_discount_vector,culture_momentum ) -> float:
        """
        Calc the individual culture of the attitudeion matrix for homophilly
        """
        av_behaviour = self.quick_av_behaviour(attitudes,threshold_weighting_array)
        av_behaviour_list = [av_behaviour]*culture_momentum
        indiv_cul = np.matmul(normalized_discount_vector, av_behaviour_list)
        return indiv_cul

    def quick_calc_culture(self,attitude_matrix,threshold_matrix):
        """
        Create culture list from the attitudeion matrix for homophilly
        """

        cul_list = []
        for i in range(len(attitude_matrix)):
            threshold_weighting_array = threshold_matrix[i]/sum(threshold_matrix[i])
            cul_list.append(self.quick_indiv_calc_culture(attitude_matrix[i],threshold_weighting_array,self.normalized_discount_array[i], self.culture_momentum_list[i]))
        return cul_list


    def generate_init_data_behaviours(self) -> tuple:

        if self.harsh_data:
            attitude_list, threshold_list = self.generate_harsh_data()
        else:
                attitude_list = [np.random.beta(self.alpha_attitude, self.beta_attitude, size=self.M) for n in range(self.N)]
                threshold_list = [np.random.beta(self.alpha_threshold, self.beta_threshold, size=self.M) for n in range(self.N)]
        
        attitude_matrix = np.asarray(attitude_list)
        threshold_matrix = np.asarray(threshold_list)
        
        culture_list = self.quick_calc_culture(attitude_matrix,threshold_matrix)#,threshold_matrix
        
        #shuffle the indexes!
        attitude_list_sorted = [x for _,x in sorted(zip(culture_list,attitude_list))]
        attitude_array_circular = self.produce_circular_list(attitude_list_sorted)
        attitude_array_circular_indexes = list(range(len(attitude_array_circular)))
        attitude_array_circular_indexes_shuffled = self.partial_shuffle(attitude_array_circular_indexes, self.shuffle_reps)
        attitude_list_sorted_shuffle = [x for _,x in sorted(zip(attitude_array_circular_indexes_shuffled,attitude_array_circular))]

        return np.asarray(attitude_list_sorted_shuffle), threshold_matrix

    def create_agent_list(self) -> list:

        individual_params = {
                "delta_t" : self.delta_t,
                "t" : self.t,
                "M": self.M,
                "save_data" : self.save_data,
                "carbon_emissions" : self.carbon_emissions,
                "phi_array": self.phi_array,
                "averaging_method": self.averaging_method,
                "compression_factor": self.compression_factor,
        }

        agent_list = [Individual(individual_params,self.attitude_matrix_init[n],self.threshold_matrix_init[n], self.normalized_discount_array[n], self.culture_momentum_list[n]) for n in range(self.N)]

        return agent_list

    def calc_behavioural_attitude_matrix(self) ->  npt.NDArray:
        behavioural_attitude_matrix = np.array([n.attitudes for n in self.agent_list])
        return behavioural_attitude_matrix

    def calc_ego_influence_degroot(self) ->  npt.NDArray:
        return np.matmul(self.weighting_matrix, self.behavioural_attitude_matrix)

    def calc_social_component_matrix(self,ego_influence: npt.NDArray) ->  npt.NDArray:
        return ego_influence + np.random.normal(loc=0, scale=self.learning_error_scale, size=(self.N, self.M))

    def calc_total_weighting_matrix_difference(self, matrix_before: npt.NDArray, matrix_after: npt.NDArray)-> float:
        difference_matrix = np.subtract(matrix_before, matrix_after)
        total_difference = (np.abs(difference_matrix)).sum()
        return total_difference

    def update_weightings(self)-> float:
        culture_list = np.array([x.culture for x in self.agent_list])

        difference_matrix = np.subtract.outer(culture_list, culture_list)
        
        alpha = np.exp(-self.confirmation_bias*np.abs(difference_matrix))

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

    def next_step(self):
        # advance a time step
        self.t += self.delta_t
        self.steps += 1

        # execute step
        self.update_individuals()

        # update network parameters for next step
        if self.alpha_change == 1.0:
                if self.save_data:
                    self.weighting_matrix,self.weighting_matrix_convergence = self.update_weightings()
                else:
                    self.weighting_matrix,__ = self.update_weightings()
        
        self.behavioural_attitude_matrix = self.calc_behavioural_attitude_matrix()
        ego_influence = self.calc_ego_influence_degroot()
        self.social_component_matrix = self.calc_social_component_matrix(ego_influence)

        if self.steps%self.compression_factor == 0 and self.save_data:
            self.total_carbon_emissions = self.calc_total_emissions()
            (
                self.average_culture,
                self.cultural_var,
                self.min_culture,
                self.max_culture,
            ) = self.calc_network_culture()
            self.green_adoption = self.calc_green_adoption()            
            self.save_data_network()
