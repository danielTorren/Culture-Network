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
        
        self.opinion_dyanmics = parameters["opinion_dynamics"]
        self.save_data = parameters["save_data"]
        self.M = int(round(parameters["M"]))
        self.N = int(round(parameters["N"]))

        self.phi_array = np.linspace(parameters["phi_list_lower"], parameters["phi_list_upper"], num=self.M)
        np.random.shuffle(self.phi_array)


        
        self.carbon_emissions = [1]*self.M#parameters[6]
        self.delta_t = parameters["delta_t"]
        self.K = int(
            round(parameters["K"])
        )  # round due to the sampling method producing floats, lets hope this works
        self.prob_rewire = parameters["prob_rewire"]
        
        self.culture_momentum = int(
            round(parameters["culture_momentum"]) / self.delta_t
        )  # round due to the sampling method producing floats, lets hope this works
        self.learning_error_scale = parameters["learning_error_scale"]

        self.discount_factor_list = np.linspace(1, 0.1, num=self.culture_momentum)

        (
            self.alpha_attract,
            self.beta_attract,
            self.alpha_threshold,
            self.beta_threshold,
        ) = (parameters["alpha_attract"], parameters["beta_attract"], parameters["alpha_threshold"], parameters["beta_threshold"])


        self.t = 0
        self.list_people = range(self.N)

        # create network
        (
            self.adjacency_matrix,
            self.weighting_matrix,
            self.network,
            self.ego_networks,
            self.neighbours_list,
        ) = self.create_weighting_matrix()

        # create indviduals
        #self.init_data_behaviours = self.generate_init_data_behaviours_alt()
        self.attract_matrix_init, self.threshold_matrix_init = self.generate_init_data_behaviours_alt()
        self.agent_list = self.create_agent_list()

        self.behavioural_attract_matrix = self.calc_behavioural_attract_matrix()#need to leave outside as its a thing being saved, why is it being saved???

        if self.opinion_dyanmics == "SELECT":
            ego_influence = self.calc_ego_influence_alt()
        elif self.opinion_dyanmics == "DEGROOT":
            ego_influence = self.calc_ego_influence_degroot()
        else:
            raise Exception("Invalid opinion dynamics model")

        self.social_component_matrix = self.calc_social_component_matrix(ego_influence)

        self.weighting_matrix,__ = self.update_weightings()  # Should I update the weighting matrix? I feel like it makes sense if its not time dependant.

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
            self.history_weighting_matrix_convergence = [
                self.weighting_matrix_convergence
            ]
            self.history_average_culture = [self.average_culture]
            self.history_min_culture = [self.min_culture]
            self.history_max_culture = [self.max_culture]

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

    def generate_init_data_behaviours(self) -> list:
        ###init_attract, init_threshold,carbon_emissions
        attract_matrix = np.asarray(
            [
                np.random.beta(self.alpha_attract, self.beta_attract, size=self.M)
                for n in self.list_people
            ]
        )
        threshold_matrix = np.asarray(
            [
                np.random.beta(self.alpha_threshold, self.beta_threshold, size=self.M)
                for n in self.list_people
            ]
        )
        init_data_behaviours = [
            [
                [attract_matrix[n][m], threshold_matrix[n][m], self.carbon_emissions[m]]
                for m in range(self.M)
            ]
            for n in self.list_people
        ]
        return init_data_behaviours

    def generate_init_data_behaviours_alt(self) -> tuple:
        ###init_attract, init_threshold,carbon_emissions
        attract_matrix = np.asarray([np.random.beta(self.alpha_attract, self.beta_attract, size=self.M) for n in self.list_people])
        threshold_matrix = np.asarray(
            [
                np.random.beta(self.alpha_threshold, self.beta_threshold, size=self.M)
                for n in self.list_people
            ]
        )
        return np.asarray(attract_matrix),np.asarray(threshold_matrix)

    def create_agent_list(self) -> list:
        agent_list = [
            Individual(
                self.attract_matrix_init[i],
                self.threshold_matrix_init[i],
                self.delta_t,
                self.culture_momentum,
                self.t,
                self.M,
                self.save_data,
                self.carbon_emissions,
                self.discount_factor_list
            )
            for i in self.list_people
        ]
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

    def update_weightings(self)-> float:
        culture_list = np.array([x.culture for x in self.agent_list])
        #print("culture_list",culture_list)
        difference_matrix = np.subtract.outer(culture_list, culture_list)
        #print("difference_matrix",difference_matrix)
        alpha = (1 - 0.5*np.abs(difference_matrix))
        #print("alpha",alpha)
        diagonal = self.adjacency_matrix*alpha
        norm_weighting_matrix = self.normlize_matrix(diagonal)

        if self.save_data:
            total_difference = self.calc_total_weighting_matrix_difference(
                self.weighting_matrix, norm_weighting_matrix
            )
            return norm_weighting_matrix,total_difference
        else:
            return norm_weighting_matrix,0#bodge for mypy

    def calc_total_emissions(self) -> int:
        return sum([x.carbon_emissions for x in self.agent_list])

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
            self.agent_list[i].t = self.t
            self.agent_list[i].next_step(self.social_component_matrix[i])

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

    def next_step(self):
        # advance a time step
        self.t += self.delta_t

        # execute step
        self.update_individuals()

        # update network parameters for next step
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
            self.save_data_network()
