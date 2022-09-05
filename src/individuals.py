from logging import raiseExceptions
import numpy.typing as npt
import numpy as np
from scipy.stats import gmean

class Individual:

    """
    Class for indivduals
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
        self.averaging_method = individual_params["averaging_method"]
        self.compression_factor = individual_params["compression_factor"]
        self.phi_array = individual_params["phi_array"]

        if self.averaging_method == "Threshold weighted arithmetic":
            self.threshold_sum = sum(init_data_thresholds)
            self.threshold_weighting_array = init_data_thresholds/self.threshold_sum

        self.values = self.attitudes - self.thresholds
        
        self.av_behaviour = self.update_av_behaviour()

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

    def update_av_behaviour(self):
        if self.averaging_method == "Arithmetic":
            return np.mean(self.attitudes)
        elif self.averaging_method == "Threshold weighted arithmetic":
            return np.matmul(self.threshold_weighting_array, self.attitudes)#/(self.M)
        else:
            raiseExceptions("Invalid averaging method choosen try: Arithmetic or Geometric")

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

        self.av_behaviour = self.update_av_behaviour()
        self.update_av_behaviour_list()

        self.culture = self.calc_culture()
        
        if self.save_data:
            self.total_carbon_emissions = self.update_total_emissions()
            if self.steps%self.compression_factor == 0:
                self.save_data_individual()
        


