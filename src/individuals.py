"""Define individual agent class
A module that defines "individuals" that have vectors of attitudes towards behaviours whose evolution
is determined through weighted social interactions.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

#imports
from logging import raiseExceptions
import numpy.typing as npt
import numpy as np

#modules
class Individual:

    """
    Class to represent individuals with identities and behaviours

    ...

    Parameters
    ----------
    individual_params: dict,
        useful parameters from the network 
    init_data_attitudes: npt.NDArray[float]
        array of inital attitudes generated previously from a beta distribution, evolves over time
    init_data_thresholds: npt.NDArray[float]
        array of inital thresholds generated previously from a beta distribution
    normalized_discount_vector: npt.NDArray[float]
        normalized single row of the discounts to individual memory when considering how the past influences current identity
    culture_momentum: int
        the number of steps into the past that are considered when calculating identity

    Attributes
    ----------
    self.values = self.attitudes - self.thresholds
    self.av_behaviour = np.mean(self.attitudes)
    self.av_behaviour_list = [self.av_behaviour]*self.culture_momentum
    self.culture = self.calc_culture()
    self.total_carbon_emissions = self.update_total_emissions()
    self.history_behaviour_values = [list(self.values)]
    self.history_behaviour_attitudes = [list(self.attitudes)]
    self.history_behaviour_thresholds = [list(self.thresholds)]
    self.history_av_behaviour = [self.av_behaviour]
    self.history_culture = [self.culture]
    self.history_carbon_emissions = [self.total_carbon_emissions]
    

    Methods
    -------
    normlize_matrix(matrix: npt.NDArray) ->  npt.NDArray: 
        Row normalize an array

    def update_av_behaviour_list(self):

    def calc_culture(self) -> float:

    def update_values(self):

    def update_attitudes(self,social_component):

    def update_total_emissions(self):


    def save_data_individual(self):

    def next_step(self, t:float,steps: int,  social_component: npt.NDArray):


    """


    def __init__(
        self, individual_params, init_data_attitudes, init_data_thresholds, normalized_discount_vector , culture_momentum
    ):
        
        self.attitudes = init_data_attitudes
        self.thresholds = init_data_thresholds
        self.normalized_discount_vector = normalized_discount_vector
        self.culture_momentum = culture_momentum
        
        self.M = individual_params["M"]
        self.t = individual_params["t"]
        self.delta_t = individual_params["delta_t"]
        self.save_data = individual_params["save_data"]
        self.carbon_intensive_list = individual_params["carbon_emissions"]
        self.compression_factor = individual_params["compression_factor"]
        self.phi_array = individual_params["phi_array"]

        self.values = self.attitudes - self.thresholds
        
        self.av_behaviour = np.mean(self.attitudes)

        self.av_behaviour_list = [self.av_behaviour]*self.culture_momentum
        self.culture = self.calc_culture()

        if self.save_data:
            self.total_carbon_emissions = self.update_total_emissions()
            self.history_behaviour_values = [list(self.values)]
            self.history_behaviour_attitudes = [list(self.attitudes)]
            self.history_behaviour_thresholds = [list(self.thresholds)]
            self.history_av_behaviour = [self.av_behaviour]
            self.history_culture = [self.culture]
            self.history_carbon_emissions = [self.total_carbon_emissions]

    def update_av_behaviour_list(self):
        self.av_behaviour_list.pop()
        self.av_behaviour_list.insert(0,self.av_behaviour)

    def calc_culture(self) -> float:
        return np.matmul(self.normalized_discount_vector, self.av_behaviour_list)#here discount list is normalized

    def update_values(self):
        self.values = self.attitudes - self.thresholds

    def update_attitudes(self,social_component):
        self.attitudes = self.attitudes*(1 - self.phi_array*self.delta_t) + self.phi_array*self.delta_t*(social_component)  
    
    def update_total_emissions(self):
        return sum(self.carbon_intensive_list[i] for i in range(self.M) if self.values[i] <= 0)

    def save_data_individual(self):
        self.history_behaviour_values.append(list(self.values))
        #print("test", list(self.attitudes)[0])
        self.history_behaviour_attitudes.append(list(self.attitudes))
        self.history_behaviour_thresholds.append(list(self.thresholds))
        self.history_culture.append(self.culture)
        self.history_av_behaviour.append(self.av_behaviour)
        self.history_carbon_emissions.append(self.total_carbon_emissions)

    def next_step(self, t:float,steps: int,  social_component: npt.NDArray):
        self.t = t
        self.steps = steps

        self.update_values()
        self.update_attitudes(social_component)

        self.av_behaviour = np.mean(self.attitudes)
        self.update_av_behaviour_list()

        self.culture = self.calc_culture()
        
        if self.save_data:
            self.total_carbon_emissions = self.update_total_emissions()
            if self.steps%self.compression_factor == 0:
                self.save_data_individual()
        


