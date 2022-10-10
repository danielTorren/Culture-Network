"""Create social network with individuals
A module that use input data to generate a social network containing individuals who each have multiple 
behaviours. The weighting of indivdiuals within the social network is determined by the identity distance 
between neighbours. The simulation evolves over time saving data at set intervals to reduce data output.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

#imports
import numpy as np
import networkx as nx
from individuals import Individual
import numpy.typing as npt
from logging import raiseExceptions

#modules
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
    save_data : bool
        whether or not to save data. Set to 0 if only interested in end state of the simulation. If 1 will save
        data into timeseries (lists or lists of lists for arrays) which can then be either accessed directly in 
        the social network object or saved into csv's
    compression_factor: int
        how often data is saved. If set to 1 its every step, then 10 is every 10th steps. Higher value gives lower 
        resolution for graphs but more managable saved or end object size
    t: float
        keep track of time, increased with each step by time step delta_t
    steps: int
        count number of steps in the simualtion
    delta_t: float
        size of time step in the simulation, default should be 1 and if this is changed then time dependant parameters
        such as phi(the degree of conspicuous consumption or social suseptability of a behaviour) also needs to be
        adjusted i.e. a larger delta_t requires a smaller phi to produce the same resolution of results
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
    present_discount_factor: float
        the degree to which a single time step in the past is worth less than the present vs immediate past. The lower this parameter
        the more emphasis is put on the present as the most important moment
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
    inverse_homophily: float
        the degree to which an agents neighbours are dis-similar to them identity wise. A value of 0 means agents are placed in the small world social network
        next to their cultural peers. The closer to 1 the more times agents are swapped around in the network using the Fisher Yates shuffle. Needs to be positive but recommended [0,1]
    homophilly_rate: float
        the greater this value the more shuffling of individuals occur for a given value of inverse_homophily
    shuffle_reps: int
        the number of time indivdiuals swap positions within the social network, note that this doesn't affect the actual network structure
    alpha_attitude,beta_attitude, alpha_threshold, beta_threshold: float
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
    behavioural_attitude_matrix: npt.NDArray[float]
        array of shape (N,M) with the values of the attitudes of each agent towards M behaviours which evolves over time
    adjacency_matrix: npt.NDArray[bool]
        array giveing social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
    weighting_matrix: npt.NDArray[float]
        an NxN array how how much each agent values the opinion of their neighbour. Note that is it not symetric and agent i doesn't need to value the 
        opinion of agent j as much as j does i's opinion
    network: networkx graph
        a networkx watts strogatz small world graph
    social_component_matrix: npt.NDArray[float]
        NxM array of influence of neighbours on an individual's attitudes towards M behaviours
    average_culture: float
        average identity of society
    cultural_var: float
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
    history_cultural_var: list[float]
        time series of cultural_var
    history_time: list[float]
        time series of time
    history_total_carbon_emission: list[float]
        time series of total_carbon_emissions in the system, not the carbon_emissions for each behaviour
    history_weighting_matrix_convergence: list[float]
        time series of weighting_matrix_convergence
    history_average_culture: list[float]
        time series of average agent culture
    history_min_culture: list[float]
        time series of min_culture
    history_green_adoption: list[float]
        time series of green_adoption
    

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
        Calculate the identity of individuals not using class properties. Used once for intial homophily measures
    generate_init_data_behaviours() -> tuple:
        Generate the intial values for agent behavioural attitudes and thresholds 
    create_agent_list() -> list:
        Create list of Individual objects that each have behaviours
    calc_behavioural_attitude_matrix() ->  npt.NDArray:
        Get NxM array of agent attitude towards M behaviours
    calc_ego_influence_alt() ->  npt.NDArray:
        Calculate the influence of neighbours by selecting a neighbour to imitate using the link strength as a probability of selection
    calc_ego_influence_degroot() ->  npt.NDArray:
        Calculate the influence of neighbours using the Degroot model of weighted aggregation
    calc_social_component_matrix() ->  npt.NDArray:
        Combine neighbour influence and social leanring error to updated individual behavioural attitudes
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
        Update Idividual objects with new information
    save_data_network():
        Save time series data
    next_step():
        Push the simulation forwards one time step
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

        #time
        self.t = 0
        self.steps = 0
        self.delta_t = parameters["delta_t"]

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
        #self.confirmation_bias = parameters["confirmation_bias"]
        self.confirmation_bias =  np.random.normal(loc=parameters["confirmation_bias"], scale=20, size=(self.N,1))
        self.learning_error_scale = parameters["learning_error_scale"]

        #social influence of behaviours
        self.phi_array = np.linspace(parameters["phi_lower"], parameters["phi_upper"], num=self.M)# CONSPICUOUS CONSUMPTION OF BEHAVIOURS - THE HIgher the more social prestige associated with this behaviour
        
        #emissions associated with each behaviour
        self.carbon_emissions = [1]*self.M #parameters["carbon_emissions"], removed for the sake of the SA

        #network homophily
        self.inverse_homophily = parameters["inverse_homophily"]#0-1
        self.homophilly_rate = parameters["homophilly_rate"]
        self.shuffle_reps = int(round((self.N**self.homophilly_rate)*self.inverse_homophily))#int(round((self.N)*self.inverse_homophily))#im going to square it
        
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
        ) = self.create_weighting_matrix()

        self.social_component_matrix = self.calc_social_component_matrix()

        if self.alpha_change != 0.0:
            self.weighting_matrix,__ = self.update_weightings() 

        if self.save_data:
            self.total_carbon_emissions = self.calc_total_emissions()
            
            # calc_netork density
            self.calc_network_density()

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
        
        return np.asarray(normalized_discount_array)

    def calc_network_density(self):
        actual_connections = self.weighting_matrix.sum()
        potential_connections = (self.N * (self.N - 1)) / 2
        network_density = actual_connections / potential_connections
        print("network_density = ", network_density)

    def create_weighting_matrix(self)-> tuple[npt.NDArray, npt.NDArray, nx.Graph]:  
        # SMALL WORLD
        ws = nx.watts_strogatz_graph(
            n=self.N, k=self.K, p=self.prob_rewire, seed=self.set_seed
        )  # Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
        weighting_matrix = nx.to_numpy_array(ws)
        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)

        return (
            weighting_matrix,
            norm_weighting_matrix,
            ws,
        )  # num_neighbours,

    def produce_circular_list(self,list) -> list:
        first_half = list[::2]
        second_half =  (list[1::2])[::-1]
        circular = first_half + second_half
        return circular

    def partial_shuffle(self, l, swap_reps) -> list:
        n = len(l)
        for _ in range(swap_reps):
            a, b = np.random.randint(low = 0, high = n, size=2)
            l[b], l[a] = l[a], l[b]
        return l

    def quick_calc_culture(self,attitude_matrix) -> list:
        """
        Create culture list from the attitudeion matrix for homophilly
        """

        cul_list = []
        for i in range(len(attitude_matrix)):
            av_behaviour = np.mean(attitude_matrix[i])
            av_behaviour_list = [av_behaviour]*self.culture_momentum_list[i]
            indiv_cul = np.matmul(self.normalized_discount_array[i], av_behaviour_list)
            cul_list.append(indiv_cul)
        return cul_list


    def generate_init_data_behaviours(self) -> tuple:

        attitude_list = [np.random.beta(self.alpha_attitude, self.beta_attitude, size=self.M) for n in range(self.N)]
        threshold_list = [np.random.beta(self.alpha_threshold, self.beta_threshold, size=self.M) for n in range(self.N)]
        
        attitude_matrix = np.asarray(attitude_list)
        threshold_matrix = np.asarray(threshold_list)
        
        culture_list = self.quick_calc_culture(attitude_matrix)#,threshold_matrix
        
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
                "compression_factor": self.compression_factor,
        }

        agent_list = [Individual(individual_params,self.attitude_matrix_init[n],self.threshold_matrix_init[n], self.normalized_discount_array[n], self.culture_momentum_list[n]) for n in range(self.N)]

        return agent_list

    def calc_behavioural_attitude_matrix(self) ->  npt.NDArray:
        behavioural_attitude_matrix = np.array([n.attitudes for n in self.agent_list])
        return behavioural_attitude_matrix

    def calc_ego_influence_alt(self) ->  npt.NDArray:
        k_list = [np.random.choice(range(self.N), 1, p=self.weighting_matrix[n])[0] for n in range(self.N)]#for each indiviudal select a neighbour using the row of the alpha matrix as the probability
        return np.array([self.agent_list[k].attitudes for k in k_list])#make a new NxM where each row is what agent n is going to learn from their selected agent k

    def calc_ego_influence_degroot(self) ->  npt.NDArray:
        return np.matmul(self.weighting_matrix, self.behavioural_attitude_matrix)

    def calc_social_component_matrix(self,) ->  npt.NDArray:
        #ego_influence = self.calc_ego_influence_degroot()
        ego_influence = self.calc_ego_influence_alt()# calc_ego_influence_alt

        return ego_influence + np.random.normal(loc=0, scale=self.learning_error_scale, size=(self.N, self.M))

    def calc_total_weighting_matrix_difference(self, matrix_before: npt.NDArray, matrix_after: npt.NDArray)-> float:
        difference_matrix = np.subtract(matrix_before, matrix_after)
        total_difference = (np.abs(difference_matrix)).sum()
        return total_difference

    def update_weightings(self)-> float:
        culture_list = np.array([x.culture for x in self.agent_list])

        difference_matrix = np.subtract.outer(culture_list, culture_list)

        alpha_numerator = np.exp(-np.multiply(self.confirmation_bias,np.abs(difference_matrix)))

        non_diagonal_weighting_matrix = self.adjacency_matrix*alpha_numerator#We want only those values that have network connections
        
        norm_weighting_matrix = self.normlize_matrix(non_diagonal_weighting_matrix)#normalize the matrix row wise

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
            np.var(culture_list),
            max(culture_list),
            min(culture_list),
        )

    def calc_green_adoption(self) -> float:
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

        #self.confirmation_bias += 0.05
        #self.confirmation_bias = self.confirmation_bias_list[self.steps]
        #print(self.confirmation_bias)

        # execute step
        self.update_individuals()

        # update network parameters for next step
        if self.alpha_change == 1.0:
                if self.save_data:
                    self.weighting_matrix,self.weighting_matrix_convergence = self.update_weightings()
                else:
                    self.weighting_matrix,__ = self.update_weightings()
        
        self.behavioural_attitude_matrix = self.calc_behavioural_attitude_matrix()
        self.social_component_matrix = self.calc_social_component_matrix()

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
