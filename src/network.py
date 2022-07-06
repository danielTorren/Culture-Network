import numpy as np
import networkx as nx
from individuals import Individual
import numpy.typing as npt

class Network:

    """
    Class for network which holds network
    Properties: Culture, Behaviours
    """

    def __init__(self, parameters:list):
        #print(parameters)
        
        self.set_seed = int(round(parameters["set_seed"]))
        np.random.seed(
            self.set_seed
        )  # not sure if i have to be worried about the randomness of the system being reproducible
        
        self.alpha_change = parameters["alpha_change"]

        self.opinion_dyanmics = parameters["opinion_dynamics"]
        self.save_data = parameters["save_data"]
        self.M = int(round(parameters["M"]))
        self.N = int(round(parameters["N"]))

        self.phi_array = np.linspace(parameters["phi_list_lower"], parameters["phi_list_upper"], num=self.M)
        np.random.shuffle(self.phi_array)

        self.carbon_emissions = parameters["carbon_emissions"]#emissions associated with each behaviour

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

        self.delta_t = parameters["delta_t"]
        self.K = int(
            round(parameters["K"])
        )  # round due to the sampling method producing floats, lets hope this works
        self.prob_rewire = parameters["prob_rewire"]
        
        self.culture_momentum_real = parameters["culture_momentum_real"]
        self.culture_momentum = int(
            round(self.culture_momentum_real/ self.delta_t)
        )  # round due to the sampling method producing floats, lets hope this works
        #print("cul mom:", parameters["culture_momentum"],self.culture_momentum )

        self.learning_error_scale = parameters["learning_error_scale"]

        self.discount_factor = parameters["discount_factor"]

        (
            self.alpha_attract,
            self.beta_attract,
            self.alpha_threshold,
            self.beta_threshold,
        ) = (parameters["alpha_attract"], parameters["beta_attract"], parameters["alpha_threshold"], parameters["beta_threshold"])


        self.t = 0
        self.list_people = range(self.N)

        # create indviduals#Do a homophilly
        self.attract_matrix_init, self.threshold_matrix_init = self.generate_init_data_behaviours()#self.generate_init_data_behaviours_homo()

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
            self.weighting_matrix,__ = self.update_weightings_alt()  # Should I update the weighting matrix? I feel like it makes sense if its not time dependant.

        self.total_carbon_emissions = self.calc_total_emissions()

        if self.save_data:
            # calc_netork density
            self.calc_network_density()

            (
                self.average_culture,
                self.cultural_var,
                self.min_culture,
                self.max_culture,
            ) = self.calc_network_culture()
            self.weighting_matrix_convergence = (
                np.nan
            )  # there is no convergence in the first step, to deal with time issues when plotting

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
        norm_matrix = matrix / row_sums[:, np.newaxis]

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

    def generate_init_data_behaviours(self) -> tuple:
        ###init_attract, init_threshold,carbon_emissions
        attract_matrix = np.asarray([np.random.beta(self.alpha_attract, self.beta_attract, size=self.M) for n in self.list_people])
        threshold_matrix = np.asarray(
            [
                np.random.beta(self.alpha_threshold, self.beta_threshold, size=self.M)
                for n in self.list_people
            ]
        )
        return np.asarray(attract_matrix),np.asarray(threshold_matrix)

    def produce_circular_list(self,list):
        first_half = list[::2]
        second_half =  (list[1::2])[::-1]
        circular = first_half + second_half
        return circular

    def generate_init_data_behaviours_homo(self) -> tuple:
        ###init_attract, init_threshold,carbon_emissions
        attract_list = [np.random.beta(self.alpha_attract, self.beta_attract, size=self.M) for n in self.list_people]
        threshold_list = [np.random.beta(self.alpha_threshold, self.beta_threshold, size=self.M) for n in self.list_people]
        attract_matrix = np.asarray(attract_list)
        threshold_matrix = np.asarray(threshold_list)

        behave_value_matrix = attract_matrix - threshold_matrix
        culture_list = behave_value_matrix.sum(axis=1)/behave_value_matrix.shape[1]


        attract_list_sorted = [x for _,x in sorted(zip(culture_list,attract_list))]
        threshold_list_sorted = [x for _,x in sorted(zip(culture_list,threshold_list))]

        attract_array_circular = np.asarray(self.produce_circular_list(attract_list_sorted))
        threshold_array_circular = np.asarray(self.produce_circular_list(threshold_list_sorted))

        return attract_array_circular,threshold_array_circular

    def create_agent_list(self) -> list:

        individual_params = {
                "delta_t" : self.delta_t,
                "culture_momentum" : self.culture_momentum,
                "t" : self.t,
                "M": self.M,
                "save_data" : self.save_data,
                "carbon_emissions" : self.carbon_emissions,
                "discount_factor" : self.discount_factor,
                "carbon_price_state" : self.carbon_price_state,
                "information_provision_state" : self.information_provision_state,
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

    def update_weightings(self)-> float:
        culture_list = np.array([x.culture for x in self.agent_list])

        #use the chosen method
        difference_matrix = np.subtract.outer(culture_list, culture_list)
        alpha = (1 - 0.5*np.abs(difference_matrix))
        diagonal = self.adjacency_matrix*alpha

        #
        norm_weighting_matrix = self.normlize_matrix(diagonal)

        if self.save_data:
            total_difference = self.calc_total_weighting_matrix_difference(
                self.weighting_matrix, norm_weighting_matrix
            )
            return norm_weighting_matrix,total_difference
        else:
            return norm_weighting_matrix,0#BODGE! bodge for mypy

    def update_weightings_alt(self)-> float:
        culture_list = np.array([x.culture for x in self.agent_list])

        #use the chosen method
        difference_matrix = np.subtract.outer(culture_list, culture_list)
        alpha = np.exp(-np.abs(difference_matrix))
        #print(alpha)
        diagonal = self.adjacency_matrix*alpha
        #print(diagonal)

        #
        norm_weighting_matrix = self.normlize_matrix(diagonal)
        #print(norm_weighting_matrix)
        

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

    def update_individuals(self):
        for i in self.list_people:
            if self.carbon_price_state:
                self.agent_list[i].carbon_price = self.carbon_price
            self.agent_list[i].next_step(self.t,self.social_component_matrix[i])

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
        #print("TIME")
        # advance a time step
        self.t += self.delta_t

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
                        self.weighting_matrix,self.weighting_matrix_convergence = self.update_weightings_alt()
                else:
                        self.weighting_matrix,__ = self.update_weightings_alt()
        
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
            self.save_data_network()
