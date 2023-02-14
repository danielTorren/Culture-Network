"""Create social network with individuals
A module that use input data to generate a social network containing individuals who each have multiple 
behaviours. The weighting of indivdiuals within the social network is determined by the identity distance 
between neighbours. The simulation evolves over time saving data at set intervals to reduce data output.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
from resources.individuals import Individual
from resources.green_individuals import Green_individual
from resources.green_fountains import Green_fountain

# modules
class Network:
    """
    Class to represent social network of simulation which is composed of individuals each with identities and behaviours

    ...

    Parameters
    ----------
    parameters : dict
        Dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters

    Attributes
    ----------
    set_seed : int
        stochastic seed of simulation for reproducibility
    alpha_change : float
        determines how  and how often agent's re-asses their connections strength in the social network
    save_timeseries_data : bool
        whether or not to save data. Set to 0 if only interested in end state of the simulation. If 1 will save
        data into timeseries (lists or lists of lists for arrays) which can then be either accessed directly in
        the social network object or saved into csv's
    compression_factor: int
        how often data is saved. If set to 1 its every step, then 10 is every 10th steps. Higher value gives lower
        resolution for graphs but more managable saved or end object size
    degroot_aggregation:
        determines whether using Degroot or voter model style social learning
    t: float
        keep track of time, increased with each step by time step delta_t
    steps: int
        count number of steps in the simualtion
    delta_t: float
        size of time step in the simulation, default should be 1 and if this is changed then time dependant parameters
        such as phi(the degree of conspicuous consumption or social suseptability of a behaviour) also needs to be
        adjusted i.e. a larger delta_t requires a smaller phi to produce the same resolution of results
    guilty_individuals: bool
        Do individuals strive to be green?
    guilty_individual_power: float
        How much does identity drive the strive to be green
    M: int
        number of behaviours per individual. These behaviours are meant to represent action decisions that operate under the
        same identity such as the decision to cycle to work or take the car.
    N: int
        number of individuals in the social network
    K: int
        number of connections per individual in the small world social network. The greater the prob_rewire the greater
        agent to agent variation there is in this number. Must be less than N.
    prob_rewire: float
        Probability of rewiring connections in the social network from one indivdual to another. The greater this values
        the more long distance connections within the network exist. Domain = [0,1]
    culture_momentum_real: float
        the real time that individuals consider in their past when evaluating their identity. The greater this is the further
        into the past individuals look.
    culture_momentum: float
        the number of steps into the past that are considered when individuals consider their identity
    culture_momentum_list: list[int]
        list of each agents cultural momentum. Allows for greater heterogenity in agent population with some being laggards and
        others being fast to change
    discount_factor: float
        the degree to which each previous time step has a decreasing importance to an individuals memory. Domain = [0,1]
    normalized_discount_array: npt.NDArray[float]
        array where each row represents the specific agent and the columns a time series that is the length of that
        agents culture_momentum. The array is row normalized so that each row sums to 1
    confirmation_bias: float
        the extent to which individuals will only pay attention to other idividuals who are similar to them in social interactions
        values of > 50 usually lead to clustering of individuals into information bubbles that dont interact very much slowing cnsensus
        formation. If set to 0 then each individual considers its neighbours opinion equally. Negative values have a similar effect
        as who the individual pays attention too bounces around so much that its the same as listening to everyone equally
    learning_error_scale: float
        the standard deviation of a guassian distribution centered on zero, representing the imperfection of learning in social transmission
    phi_array: npt.NDArray[float]
        list of degree of social susceptibility or conspicous consumption of the different behaviours. Affects how much social interaction
        influences an individuals attitude towards a behaviour. As the behaviours are abstract the currnet values
        are not based on any empircal data hence the network behaviour is independent of their order so the
        list is generated using a linspace function using input parameters phi_array_lower and phi_array_upper. each element has domain = [0,1]
    carbon_emissions: list
        list of emissions of each behaviour, defaulted to 1 for each behaviour if its performed in a brown way, B =< 0
    action_observation: float
        Do actions matter more than attitudes in opinion formation
    homophily: float
        the degree to which an agents neighbours are similar to them identity wise. A value of 1 means agents are placed in the small world social network
        next to their cultural peers. The closer to 0 the more times agents are swapped around in the network using the Fisher Yates shuffle. Domain [0,1]
    homophilly_rate: float
        the greater this value the more shuffling of individuals occur for a given value of homophily
    shuffle_reps: int
        the number of time indivdiuals swap positions within the social network, note that this doesn't affect the actual network structure
    a_attitude,bb_attitude, a_threshold, b_threshold: float
        parameters used to generate the beta distribution for the intial agent attitudes and threholds for each behaviour respectively.
        The same distribution is used for all agents adn behaviours
    attitude_matrix_init: npt.NDArray[float]
        array of shape (N,M) with the intial values of the attitudes of each agent towards M behaviours which then evolve over time. Used to generated
        Indivdual objects
    threshold_matrix_init: npt.NDArray[float]
        array of shape (N,M) with the intial values of the thresholds of each agent towards M behaviours, these are static. Used to generated
        Indivdual objects
    agent_list: list[Individual]
        list of Individuals objects containing behaviours of each individual
    adjacency_matrix: npt.NDArray[bool]
        array giveing social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
    weighting_matrix: npt.NDArray[float]
        an NxN array how how much each agent values the opinion of their neighbour. Note that is it not symetric and agent i doesn't need to value the
        opinion of agent j as much as j does i's opinion
    network: nx.Graph
        a networkx watts strogatz small world graph
    social_component_matrix: npt.NDArray[float]
        NxM array of influence of neighbours on an individual's attitudes towards M behaviours
    average_culture: float
        average identity of society
    std_culture : float
        standard deviation of agent identies at time t
    var_culture: float
        variance of agent identies at time t
    min_culture: float
        minimum individual identity at time t
    max_culture: float
        maximum individual idenity at time t
    weighting_matrix_convergence: float
        total change in agent link strength from previous to current step, a measure of convergece should tend to zero
    green_adoption: float
        what percentage of total behaviours are green, i.e. B > 0
    total_carbon_emissions: float
        total emissions due to behavioural choices of agents. Note the difference between this and carbon_emissions list for each behaviour
    history_weighting_matrix: list[npt.NDArray[float]]
        time series of weighting_matrix
    history_social_component_matrix: list[npt.NDArray[float]]
        time series of social_component_matrix
    history_var_culture: list[float]
        time series of var_culture
    history_time: list[float]
        time series of time
    history_total_carbon_emission: list[float]
        time series of total_carbon_emissions in the system, not the carbon_emissions for each behaviour
    history_weighting_matrix_convergence: list[float]
        time series of weighting_matrix_convergence
    history_average_culture: list[float]
        time series of average agent culture
    history_std_culture: list[float]
        time series of std_culture
    history_min_culture: list[float]
        time series of min_culture
    history_green_adoption: list[float]
        time series of green_adoption
    prop_green: float
        proportion of network that are green emitters
    


    Methods
    -------
    normlize_matrix(matrix: npt.NDArray) ->  npt.NDArray:
        Row normalize an array
    calc_normalized_discount_array(self):
        Returns row normalized discount array
    calc_network_density():
        Prints social network density
    create_weighting_matrix()-> tuple[npt.NDArray, npt.NDArray, nx.Graph]:
        Create small world social network
    produce_circular_list(list) -> list:
        Makes an ordered list circular so that the start and end values are close in value
    partial_shuffle(l, swap_reps) -> list:
        Partially shuffle a list using Fisher Yates shuffle
    quick_calc_culture(attitude_matrix) -> list:
        Calculate the identity of individuals not using class properties. Used once for initial homophily measures
    generate_init_data_behaviours() -> tuple:
        Generate the initial values for agent behavioural attitudes and thresholds
    create_agent_list() -> list:
        Create list of Individual objects that each have behaviours
    calc_ego_influence_voter() ->  npt.NDArray:
        Calculate the influence of neighbours by selecting a neighbour to imitate using the link strength as a probability of selection
    calc_ego_influence_degroot() ->  npt.NDArray:
        Calculate the influence of neighbours using the Degroot model of weighted aggregation
    calc_social_component_matrix() ->  npt.NDArray:
        Combine neighbour influence and social learning error to updated individual behavioural attitudes
    calc_total_weighting_matrix_difference(matrix_before: npt.NDArray, matrix_after: npt.NDArray)-> float:
        Calculate the total change in link strength over one time step
    update_weightings()-> float:
        Update the link strength array according to the new agent identities
    calc_total_emissions() -> int:
        Calculate total carbon emissions of N*M behaviours
    calc_network_culture() ->  tuple[float, float, float, float]:
        Return various identity properties
    calc_green_adoption() -> float:
        Calculate the percentage of green behaviours adopted
    update_individuals():
        Update Individual objects with new information
    save_timeseries_data_network():
        Save time series data
    next_step():
        Push the simulation forwards one time step
    """

    def __init__(self, parameters: list):
        """
        Constructs all the necessary attributes for the Network object.

        Parameters
        ----------
        parameters : dict
            Dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters

        """

        self.set_seed = parameters["set_seed"]
        np.random.seed(self.set_seed)

        self.network_structure = parameters["network_structure"]
        if self.network_structure == "small_world":
            self.K = int(round(parameters["K"]))  # round due to the sampling method producing floats in the Sobol Sensitivity Analysis (SA)
            self.prob_rewire = parameters["prob_rewire"]
        elif self.network_structure == "erdos_renyi_graph":
            self.prob_edge = parameters["prob_edge"]
        elif self.network_structure == "barabasi_albert_graph":
            self.k_new_node = int(round(parameters["k_new_node"]))#Number of edges to attach from a new node to existing nodes


        self.alpha_change = parameters["alpha_change"]
        self.save_timeseries_data = parameters["save_timeseries_data"]
        self.compression_factor = parameters["compression_factor"]
        self.degroot_aggregation = parameters["degroot_aggregation"]
        self.immutable_green_fountains = parameters["immutable_green_fountains"]
        self.polarisation_test = parameters["polarisation_test"]#whether to have a = b

        self.guilty_individuals = parameters["guilty_individuals"]
        self.guilty_individual_power = parameters["guilty_individual_power"]
        self.moral_licensing = parameters["moral_licensing"]

        # time
        self.t = 0
        self.steps = 0
        self.delta_t = parameters["delta_t"]

        # network
        self.green_N = int(round(parameters["green_N"]))
        if self.green_N > 0:
            self.additional_greens = parameters["additional_greens"]
        self.M = int(round(parameters["M"]))
        self.N = int(round(parameters["N"]))
        #if self.green_N > 0 and self.additional_greens:
        #    self.N = int(round(parameters["N"])) + self.green_N
        #else:
            #self.N = int(round(parameters["N"]))
        

        # culture
        self.culture_momentum_real = parameters["culture_momentum_real"]
        self.culture_momentum = int(
            round(self.culture_momentum_real / self.delta_t)
        )  # round due to the sampling method producing floats, lets hope this works

        self.culture_momentum_list = [self.culture_momentum] * self.N

        # time discounting
        self.discount_factor = parameters["discount_factor"]
        self.normalized_discount_array = self.calc_normalized_discount_array()

        # social learning and bias
        self.confirmation_bias = parameters["confirmation_bias"]
        #self.confirmation_bias = np.random.normal(
        #    loc=parameters["confirmation_bias"], scale=20, size=(self.N, 1)
        #)
        self.learning_error_scale = parameters["learning_error_scale"]

        # social influence of behaviours
        self.phi_lower = parameters["phi_lower"]
        self.phi_upper = parameters["phi_upper"]
        self.phi_array = np.linspace(self.phi_lower, self.phi_upper, num=self.M)

        # emissions associated with each behaviour
        self.carbon_intensive_list = [1] * self.M

        #how much are individuals influenced by what they see or hear, do opinions or actions more
        self.action_observation_I = parameters["action_observation_I"]
        self.action_observation_S = parameters["action_observation_S"]#["action_observation_S"]

        # network homophily
        self.homophily = parameters["homophily"]  # 0-1
        self.homophilly_rate = parameters["homophilly_rate"]
        self.shuffle_reps = int(
            round((self.N**self.homophilly_rate) * (1 - self.homophily))
        )

        # create network
        if self.green_N>0 and self.additional_greens:
            (
                self.adjacency_matrix,
                self.weighting_matrix,
                self.network,
            ) = self.create_weighting_matrix_add_greens()
        else:
            (
                self.adjacency_matrix,
                self.weighting_matrix,
                self.network,
            ) = self.create_weighting_matrix()

        self.network_density = nx.density(self.network)
        #print("self.network_density",self.network_density)

        #############################################################################################################################
        #Associate people with attitude values CHANGE THIS BACK
        
        self.a_attitude = parameters["a_attitude"]
        if self.polarisation_test:
            self.b_attitude = self.a_attitude
        else:
            #self.b_attitude = 2 - self.a_attitude 
            #self.b_attitude = self.a_attitude 
            self.b_attitude = parameters["b_attitude"]
        self.a_threshold = parameters["a_threshold"]
        self.b_threshold = parameters["b_threshold"]
        #################################################################################################################################

        (
            self.attitude_matrix_init,
            self.threshold_matrix_init,
        ) = self.generate_init_data_behaviours()#self.generate_init_data_behaviours_two_types()#
        
        self.agent_list = self.create_agent_list()

        #print("GREEN N", self.green_N)
        #print("BEFORE",len(self.agent_list),self.calc_total_emissions())

        #add the GREEN HERE !!
        if self.green_N > 0:
            if self.additional_greens:
                self.add_green_fountains_list()
                self.N = len(self.agent_list)
            else:
                if self.immutable_green_fountains:
                    self.mix_in_green_fountains()
                else:
                    self.mix_in_green_individuals()
        
        #print("AFTER",len(self.agent_list),self.calc_total_emissions())

        self.shuffle_agent_list()#partial shuffle of the list based on culture

        if self.alpha_change == 2.0:#independant behaviours
            self.weighting_matrix_list = [self.weighting_matrix]*self.M

        self.social_component_matrix = self.calc_social_component_matrix()

        if self.alpha_change != (0.0 or 2.0):
            #print("1 or 1.5")
            self.weighting_matrix, self.total_identity_differences,__ = self.update_weightings()
        elif self.alpha_change == 2.0:#independaent behaviours
            #print("2")
            self.weighting_matrix_list = self.update_weightings_list()

        self.init_total_carbon_emissions  = self.calc_total_emissions()
        self.total_carbon_emissions = self.init_total_carbon_emissions

        (
                self.culture_list,
                self.average_culture,
                self.std_culture,
                self.var_culture,
                self.min_culture,
                self.max_culture,
        ) = self.calc_network_culture()
        self.green_adoption = self.calc_green_adoption()

        if self.save_timeseries_data:
            self.var_first_behaviour = self.calc_var_first_behaviour()

            self.weighting_matrix_convergence = 0  # there is no convergence in the first step, to deal with time issues when plotting

            

            self.history_weighting_matrix = [self.weighting_matrix]
            self.history_social_component_matrix = [self.social_component_matrix]
            self.history_time = [self.t]
            
            self.history_weighting_matrix_convergence = [
                self.weighting_matrix_convergence
            ]
            self.history_average_culture = [self.average_culture]
            self.history_std_culture = [self.std_culture]
            self.history_var_culture = [self.var_culture]
            self.history_min_culture = [self.min_culture]
            self.history_max_culture = [self.max_culture]
            self.history_green_adoption = [self.green_adoption]
            self.history_total_carbon_emissions = [self.total_carbon_emissions]
            if self.alpha_change != (0.0 or 2.0):
                self.history_total_identity_differences = [self.total_identity_differences]

    def normlize_matrix(self, matrix: npt.NDArray) -> npt.NDArray:
        """
        Row normalize an array

        Parameters
        ----------
        matrix: npt.NDArrayf
            array to be row normalized

        Returns
        -------
        norm_matrix: npt.NDArray
            row normalized array
        """
        row_sums = matrix.sum(axis=1)
        norm_matrix = matrix / row_sums[:, np.newaxis]

        return norm_matrix

    def calc_normalized_discount_array(self) -> npt.NDArray:
        """
        Row normalize an array

        Parameters
        ----------
        None

        Returns
        -------
        normalized_discount_array: npt.NDArray
            row normalized truncated quasi-hyperbolic discount array
        """
        normalized_discount_array = []
        for i in self.culture_momentum_list:
            discount_row = []
            for v in range(i):
                discount_row.append(
                    (self.discount_factor) ** (self.delta_t * v)
                )
            discount_row[0] = 1.0

            normalized_discount_row = (
                np.asarray(discount_row) / sum(discount_row)
            ).tolist()

            normalized_discount_array.append(normalized_discount_row)

        return np.asarray(normalized_discount_array)

    def create_weighting_matrix(self) -> tuple[npt.NDArray, npt.NDArray, nx.Graph]:
        """
        Create watts-strogatz small world graph using Networkx library

        Parameters
        ----------
        None

        Returns
        -------
        weighting_matrix: npt.NDArray[bool]
            adjacency matrix, array giving social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
        norm_weighting_matrix: npt.NDArray[float]
            an NxN array how how much each agent values the opinion of their neighbour. Note that is it not symetric and agent i doesn't need to value the
            opinion of agent j as much as j does i's opinion
        ws: nx.Graph
            a networkx watts strogatz small world graph
        """

        if self.network_structure == "small_world":
            G = nx.watts_strogatz_graph(n=self.N, k=self.K, p=self.prob_rewire, seed=self.set_seed)  # Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
        elif self.network_structure == "erdos_renyi_graph":
            G = nx.erdos_renyi_graph(n = self.N, p = self.prob_edge, seed=self.set_seed)
        elif self.network_structure == "barabasi_albert_graph":
            G = nx.barabasi_albert_graph(n = self.N, m = self.k_new_node, seed=self.set_seed)

        #nx.draw(G)

        weighting_matrix = nx.to_numpy_array(G)

        #print(self.network_structure, weighting_matrix)

        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)

        return (
            weighting_matrix,
            norm_weighting_matrix,
            G,
        )
    
    def create_weighting_matrix_add_greens(self):
        """
        Create watts-strogatz small world graph using Networkx library but with N GREEN indivduals

        Parameters
        ----------
        None

        Returns
        -------
        weighting_matrix: npt.NDArray[bool]
            adjacency matrix, array giving social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
        norm_weighting_matrix: npt.NDArray[float]
            an NxN array how how much each agent values the opinion of their neighbour. Note that is it not symetric and agent i doesn't need to value the
            opinion of agent j as much as j does i's opinion
        ws: nx.Graph
            a networkx watts strogatz small world graph
        """

        if self.network_structure == "small_world":
            G = nx.watts_strogatz_graph(n=self.N+self.green_N, k=self.K, p=self.prob_rewire, seed=self.set_seed)  # Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
        elif self.network_structure == "erdos_renyi_graph":
            G = nx.erdos_renyi_graph(n = self.N+self.green_N, p = self.prob_edge, seed=self.set_seed)
        elif self.network_structure == "barabasi_albert_graph":
            G = nx.barabasi_albert_graph(n = self.N+self.green_N, m = self.k_new_node, seed=self.set_seed)

        #nx.draw(G)

        weighting_matrix = nx.to_numpy_array(G)

        #print(self.network_structure, weighting_matrix)

        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)

        return (
            weighting_matrix,
            norm_weighting_matrix,
            G,
        )

    def circular_agent_list(self) -> list:
        """
        Makes an ordered list circular so that the start and end values are matched in value and value distribution is symmetric

        Parameters
        ----------
        list: list
            an ordered list e.g [1,2,3,4,5]
        Returns
        -------
        circular: list
            a circular list symmetric about its middle entry e.g [1,3,5,4,2]
        """

        first_half = self.agent_list[::2]  # take every second element in the list, even indicies
        second_half = (self.agent_list[1::2])[::-1]  # take every second element , odd indicies
        self.agent_list = first_half + second_half

    def partial_shuffle_agent_list(self) -> list:
        """
        Partially shuffle a list using Fisher Yates shuffle
        """

        for _ in range(self.shuffle_reps):
            a, b = np.random.randint(
                low=0, high=self.N, size=2
            )  # generate pair of indicies to swap
            self.agent_list[b], self.agent_list[a] = self.agent_list[a], self.agent_list[b]

    def generate_init_data_behaviours(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Generate the initial values for agent behavioural attitudes and thresholds using Beta distribution

        Parameters
        ----------
        None

        Returns
        -------
        attitude_matrix: npt.NDArray
            NxM array of behavioural attitudes
        threshold_matrix: npt.NDArray
            NxM array of behavioural thresholds, represents the barriers to entry to performing a behaviour e.g the distance of a
            commute or disposable income of an individual
        """

        attitude_list = [
            np.random.beta(self.a_attitude, self.b_attitude, size=self.M)
            for n in range(self.N)
        ]

        #print("init values beahviorus", [np.mean(np.asarray(attitude_list)[:,m]) for m in range(self.M)])

        threshold_list = [
            np.random.beta(self.a_threshold, self.b_threshold, size=self.M)
            for n in range(self.N)
        ]

        attitude_matrix = np.asarray(attitude_list)
        threshold_matrix = np.asarray(threshold_list)

        return attitude_matrix, threshold_matrix

    def create_agent_list(self) -> list[Individual]:
        """
        Create list of Individual objects that each have behaviours

        Parameters
        ----------
        None

        Returns
        -------
        agent_list: list[Individual]
            List of Individual objects representing specific agents in a social network each with behavioural opinions and environmental identity
        """

        individual_params = {
            "delta_t": self.delta_t,
            "t": self.t,
            "M": self.M,
            "save_timeseries_data": self.save_timeseries_data,
            "carbon_intensive_list": self.carbon_intensive_list,
            "phi_array": self.phi_array,
            "compression_factor": self.compression_factor,
            "action_observation_I": self.action_observation_I,
            "guilty_individuals":self.guilty_individuals,
            "guilty_individual_power":self.guilty_individual_power,
            "moral_licensing": self.moral_licensing,
            "alpha_change": self.alpha_change,
        }

        agent_list = [
            Individual(
                individual_params,
                self.attitude_matrix_init[n],
                self.threshold_matrix_init[n],
                self.normalized_discount_array[n],
                self.culture_momentum_list[n],
                n
            )
            for n in range(self.N)
        ]

        return agent_list

    def mix_in_green_individuals(self):
        individual_params = {
            "delta_t": self.delta_t,
            "t": self.t,
            "M": self.M,
            "save_timeseries_data": self.save_timeseries_data,
            "carbon_intensive_list": self.carbon_intensive_list,
            "phi_array": self.phi_array,
            "compression_factor": self.compression_factor,
            "action_observation_I": self.action_observation_I,
            "guilty_individuals":self.guilty_individuals,
            "guilty_individual_power":self.guilty_individual_power,
            "moral_licensing": self.moral_licensing,
        }
        #randomly mix in the greens 
        n_list_green = np.random.choice(self.N, self.green_N,  replace=False)
        #print(n_list_green)
        for i in n_list_green:
            self.agent_list[i] = Green_individual(
                individual_params,
                self.normalized_discount_array[i],
                self.culture_momentum_list[i],)

    def mix_in_green_fountains(self):
        individual_params = {
            "M": self.M,
            "save_timeseries_data": self.save_timeseries_data,
            "carbon_intensive_list": self.carbon_intensive_list,
            "compression_factor": self.compression_factor,
        }

        #randomly mix in the greens 
        n_list_green = np.random.choice(self.N, self.green_N,  replace=False)
        #print("n to replace",n_list_green)


        for i in n_list_green:
            self.agent_list[i] = Green_fountain(individual_params)
        
    def add_green_fountains_list(self):
        individual_params = {
            "M": self.M,
            "save_timeseries_data": self.save_timeseries_data,
            "carbon_intensive_list": self.carbon_intensive_list,
            "compression_factor": self.compression_factor,
        }

        green_fountains_list = [Green_fountain(individual_params) for n in range(self.green_N)]
        self.agent_list.extend(green_fountains_list)
        #print("added", self.agent_list)


    def shuffle_agent_list(self): 
        #make list cirucalr then partial shuffle it

        self.agent_list.sort(key=lambda x: x.culture)#sorted by culture
        self.circular_agent_list()#agent list is now circular in terms of culture
        self.partial_shuffle_agent_list()#partial shuffle of the list

    def calc_ego_influence_voter(self) -> npt.NDArray:
        """
        Calculate the influence of neighbours by selecting a neighbour to imitate using the network weighting link strength as a probability of selection

        Parameters
        ----------
        None

        Returns
        -------
        neighbour_influence: npt.NDArray
            NxM array where each row represents a single interaction between Individuals that listen to a chosen neighbour
        """
        k_list = [
            np.random.choice(range(self.N), 1, p=self.weighting_matrix[n])[0]
            for n in range(self.N)
        ]  # for each individual select a neighbour using the row of the alpha matrix as the probability
        neighbour_influence = np.array(
            [((1-self.action_observation_S)*self.agent_list[k].attitudes  + self.action_observation_S*((self.agent_list[k].values + 1)/2) ) for k in k_list]
        )  # make a new NxM where each row is what agent n is going to learn from their selected agent k
        return neighbour_influence

    def calc_ego_influence_degroot(self) -> npt.NDArray:
        """
        Calculate the influence of neighbours using the Degroot model of weighted aggregation

        Parameters
        ----------
        None

        Returns
        -------
        neighbour_influence: npt.NDArray
            NxM array where each row represents the influence of an Individual listening to its neighbours regarding their
            behavioural attitude opinions, this influence is weighted by the weighting_matrix
        """

        behavioural_attitude_matrix = np.array( [((1-self.action_observation_S)*n.attitudes  + self.action_observation_S*((n.values + 1)/2) ) for n in self.agent_list] )

        neighbour_influence = np.matmul(self.weighting_matrix, behavioural_attitude_matrix)
        
        return neighbour_influence

    def calc_ego_influence_degroot_independent(self) -> npt.NDArray:
        """
        Calculate the influence of neighbours using the Degroot model of weighted aggregation, BEHAVIOURS INDEPENDANT

        Parameters
        ----------
        None

        Returns
        -------
        neighbour_influence: npt.NDArray
            NxM array where each row represents the influence of an Individual listening to its neighbours regarding their
            behavioural attitude opinions, this influence is weighted by the weighting_matrix
        """

        behavioural_attitude_matrix = np.array( [((1-self.action_observation_S)*n.attitudes  + self.action_observation_S*((n.values + 1)/2) ) for n in self.agent_list] )
        #print("behavioural_attitude_matrix",behavioural_attitude_matrix)

        #print("init values beahviorus", [np.mean(behavioural_attitude_matrix[:,m]) for m in range(self.M)])

        neighbour_influence = np.zeros((self.N, self.M))

        for m in range(self.M):
            #print("np.matmul(self.weighting_matrix_list[m], behavioural_attitude_matrix[:,m])",np.matmul(self.weighting_matrix_list[m], behavioural_attitude_matrix[:,m]))
            #print("weightign matrixs",len(self.weighting_matrix_list),self.weighting_matrix_list[m].shape )
            neighbour_influence[:, m] = np.matmul(self.weighting_matrix_list[m], behavioural_attitude_matrix[:,m])

        #print("neighbour_influence",neighbour_influence)
        #quit()
        return neighbour_influence

    def calc_social_component_matrix(self) -> npt.NDArray:
        """
        Combine neighbour influence and social learning error to updated individual behavioural attitudes

        Parameters
        ----------
        None

        Returns
        -------
        social_influence: npt.NDArray
            NxM array giving the influence of social learning from neighbours for that time step
        """

        if self.alpha_change == 2.0:
            if self.degroot_aggregation:
                ego_influence = self.calc_ego_influence_degroot_independent()
            else:
                ego_influence = self.calc_ego_influence_degroot_independent()#self.calc_ego_influence_voter_independent()
        else:
            if self.degroot_aggregation:
                ego_influence = self.calc_ego_influence_degroot()
            else:
                ego_influence = self.calc_ego_influence_voter()            

        social_influence = ego_influence + np.random.normal(
            loc=0, scale=self.learning_error_scale, size=(self.N, self.M)
        )
        return social_influence

    def calc_total_weighting_matrix_difference(
        self, matrix_before: npt.NDArray, matrix_after: npt.NDArray
    ) -> float:
        """
        Calculate the total change in link strength over one time step. Meant to show how the weighting quickly converges

        Parameters
        ----------
        matrix_before: npt.NDArray
            NxN array of agent opinion weightings from the previous time step
        matrix_after: npt.NDArray
            NxN array of agent opinion weightings from the current time step

        Returns
        -------
        total_difference: float
            total element wise difference between the arrays
        """
        difference_matrix = np.subtract(matrix_before, matrix_after)
        total_difference = (np.abs(difference_matrix)).sum()
        return total_difference

    def update_weightings(self) -> tuple[npt.NDArray, float]:
        """
        Update the link strength array according to the new agent identities

        Parameters
        ----------
        None

        Returns
        -------
        norm_weighting_matrix: npt.NDArray
            Row normalized weighting array giving the strength of inter-Individual connections due to similarity in identity
        total_difference: float
            total element wise difference between the previous weighting arrays
        """
        culture_list = np.array([x.culture for x in self.agent_list])

        difference_matrix = np.subtract.outer(culture_list, culture_list)

        alpha_numerator = np.exp(
            -np.multiply(self.confirmation_bias, np.abs(difference_matrix))
        )

        non_diagonal_weighting_matrix = (
            self.adjacency_matrix * alpha_numerator
        )  # We want only those values that have network connections

        norm_weighting_matrix = self.normlize_matrix(
            non_diagonal_weighting_matrix
        )  # normalize the matrix row wise

        #for total_identity_differences
        difference_matrix_real_connections = abs(self.adjacency_matrix * difference_matrix)
        total_identity_differences = difference_matrix_real_connections.sum(axis=1)

        if self.save_timeseries_data:
            total_difference = self.calc_total_weighting_matrix_difference(
                self.weighting_matrix, norm_weighting_matrix
            )
            return norm_weighting_matrix, total_identity_differences, total_difference 
        else:
            return norm_weighting_matrix, total_identity_differences, 0 # BODGE! bodge for mypy
    
    def update_weightings_list(self):
        """
        Update the link strength array according to the agent ATTITUDES NOT IDENTITIES, RETURN LIST OF MATRICIES

        Parameters
        ----------
        None

        Returns
        -------
        norm_weighting_matrix: npt.NDArray
            Row normalized weighting array giving the strength of inter-Individual connections due to similarity in identity
        total_difference: float
            total element wise difference between the previous weighting arrays
        """
        weighting_matrix_list = []

        for m in range(self.M):

            attitude_star_list = np.array([x.attitudes_star[m] for x in self.agent_list])

            difference_matrix = np.subtract.outer(attitude_star_list, attitude_star_list)
            #print("difference_matrix",difference_matrix)

            alpha_numerator = np.exp(
                -np.multiply(self.confirmation_bias, np.abs(difference_matrix))
            )

            non_diagonal_weighting_matrix = (
                self.adjacency_matrix * alpha_numerator
            )  # We want only those values that have network connections

            norm_weighting_matrix = self.normlize_matrix(
                non_diagonal_weighting_matrix
            )  # normalize the matrix row wise

            weighting_matrix_list.append(norm_weighting_matrix)

        #print("weighting_matrix_list", weighting_matrix_list[0], weighting_matrix_list[2])

        return weighting_matrix_list


    def calc_total_emissions(self) -> int:
        """
        Calculate total carbon emissions of N*M behaviours

        Parameters
        ----------
        None

        Returns
        -------
        total_network_emissions: float
            total network emissions from each Individual object
        """
        total_network_emissions = sum(
            [x.total_carbon_emissions for x in self.agent_list]
        )
        return total_network_emissions

    def calc_network_culture(self) -> tuple[float, float, float, float]:
        """
        Return various identity properties, such as mean, variance, min and max

        Parameters
        ----------
        None

        Returns
        -------
        culture_mean: float
            mean of network identity at time step t
        culture_variance: float
            variance of network identity at time step t
        culture_max: float
            max of network identity at time step t
        culture_min: float
            min of network identity at time step t
        """
        culture_list = [x.culture for x in self.agent_list]
        culture_mean = np.mean(culture_list)
        culture_std = np.std(culture_list)
        culture_variance = np.var(culture_list)
        culture_max = max(culture_list)
        culture_min = min(culture_list)
        return (culture_list,culture_mean, culture_std, culture_variance, culture_max, culture_min)

    def calc_green_adoption(self) -> float:
        """
        Calculate the percentage of green behaviours adopted

        Parameters
        ----------
        None

        Returns
        -------
        adoption_percentage: float
            adoption percentage of green behavioural alternatives where Individuals must have behavioural value B > 0
        """
        adoption = 0
        for n in self.agent_list:
            for m in range(self.M):
                if n.values[m] > 0:
                    adoption += 1
        adoption_ratio = adoption / (self.N * self.M)
        adoption_percentage = adoption_ratio * 100
        return adoption_percentage

    def update_individuals(self):
        """
        Update Individual objects with new information regarding social interactions

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for i in range(self.N):
            self.agent_list[i].next_step(
                self.t, self.steps, self.social_component_matrix[i]
            )

    def calc_var_first_behaviour(self):
        first_m_attitude_list = [x.attitudes[0] for x in self.agent_list]
        return np.var(first_m_attitude_list)

    def save_timeseries_data_network(self):
        """
        Save time series data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.history_time.append(self.t)
        self.history_weighting_matrix.append(self.weighting_matrix)
        self.history_social_component_matrix.append(self.social_component_matrix)
        self.history_weighting_matrix_convergence.append(
            self.weighting_matrix_convergence
        )
        self.history_average_culture.append(self.average_culture)
        self.history_std_culture.append(self.std_culture)
        self.history_var_culture.append(self.var_culture)
        self.history_min_culture.append(self.min_culture)
        self.history_max_culture.append(self.max_culture)
        self.history_green_adoption.append(self.green_adoption)
        self.history_total_carbon_emissions.append(self.total_carbon_emissions)
        if self.alpha_change != (0.0 or 2.0):
            self.history_total_identity_differences.append(self.total_identity_differences)

    def next_step(self):
        """
        Push the simulation forwards one time step. First advance time, then update individuals with data from previous timestep
        then produce new data and finally save it.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # advance a time step
        self.t += self.delta_t
        self.steps += 1

        # self.confirmation_bias += 0.05
        # self.confirmation_bias = self.confirmation_bias_list[self.steps]
        # print(self.confirmation_bias)

        # execute step
        self.update_individuals()

        # update network parameters for next step
        if self.alpha_change == 1.0:
            #print("1.0")
            if self.save_timeseries_data:
                (
                    self.weighting_matrix,
                    self.total_identity_differences,
                    self.weighting_matrix_convergence,
                ) = self.update_weightings()
            else:
                self.weighting_matrix, self.total_identity_differences,__ = self.update_weightings()
        elif self.alpha_change == 2.0:#independaent behaviours
            #print("2.0 update")
            self.weighting_matrix_list = self.update_weightings_list()

        self.social_component_matrix = self.calc_social_component_matrix()
        self.total_carbon_emissions = self.calc_total_emissions()
        (
                self.culture_list,
                self.average_culture,
                self.std_culture,
                self.var_culture,
                self.min_culture,
                self.max_culture,
        ) = self.calc_network_culture()
        
        if (self.steps % self.compression_factor == 0) and (self.save_timeseries_data):

            self.var_first_behaviour = self.calc_var_first_behaviour()
            self.green_adoption = self.calc_green_adoption()
            self.save_timeseries_data_network()
