import numpy as np

class Behaviour():
    """ 
        Class for behaviours
        Properties: behaviour name, value, attract, threshold
    """
    def __init__(self, init_attract, init_threshold,behaviour_cap, carbon_emissions,attract_individual_learning,psi):
        
        self.attract = init_attract
        self.threshold = init_threshold
        self.value = self.calc_behaviour()
        self.carbon_emissions = carbon_emissions
        self.attract_individual_learning = attract_individual_learning
        self.performance = 0# AT SOME POINT THIS SHOUDL BE SET RADNOMLY
        self.individual_learning = 0# AT SOME POINT THIS SHOUDL BE SET RADNOMLY

        self.psi = psi

        self.behaviour_cap = behaviour_cap

        self.history_value = [self.value]
        self.history_attract = [self.attract]
        self.history_threshold= [self.threshold]
        self.history_performance = [self.performance]

    def calc_behaviour(self):
        value = self.attract - self.threshold 
        return value

    def update_behaviour(self):
        self.value = self.calc_behaviour()
        #print("Behavioural value",self.value)

    def perform_behaviour(self):
        """
        #TRYING TO USE LOGIT MODEL
        prob_performance = 1/(1 + np.exp(-self.value))
        self.performance = np.random.binomial(1, prob_performance)
        """
        if self.value > 0:
            self.performance = 1
        else:
            self.performance = 0

    def update_attract(self,social_component_behaviours,attract_cultural_group,information_provision,delta_t):
            
            social_learing = (social_component_behaviours - self.attract) 
            individual_learning = self.individual_learning 
            conformity_bias =  (attract_cultural_group - self.attract) 
            information_provision = information_provision 

            total = social_learing  + individual_learning #+ conformity_bias + information_provision
            #print("PROPORTIONS (S,IL,CG,IP):", social_learing, individual_learning)#,conformity_bias/total,information_provision/total

            #print("attract components!",social_learing,individual_learning,conformity_bias,information_provision)
                #social_learing + individual_learning + 
            self.attract += delta_t*(conformity_bias) #  social_learing + individual_learning + conformity_bias + information_provision

    def update_threshold(self, carbon_price_gradient):
        #print(self.threshold , carbon_price_gradient*self.carbon_emissions)
        if self.threshold < carbon_price_gradient*self.carbon_emissions:
            self.threshold = 0 
        else:
            self.threshold += -carbon_price_gradient*self.carbon_emissions

    def calc_individual_learning(self):

        #print("self.performance: ",self.performance)
        #print("inside",self.attract_individual_learning - self.attract)

        if self.performance:
            #print("1:", self.performance,self.attract,self.psi*(self.attract_individual_learning - self.attract))
            individual_learning = self.psi*(self.attract_individual_learning - self.attract)
        else:
            #print("0:", self.performance,self.value,self.threshold,self.attract,self.psi*(self.attract_individual_learning - self.attract))
            individual_learning = -self.psi*(self.attract)
        
        #print(self.psi,self.attract_individual_learning - self.attract)

        return individual_learning

    def update_individual_learning(self):
        self.individual_learning = self.calc_individual_learning()


    def save_data_behaviour(self):
        self.history_value.append(self.value)
        self.history_attract.append(self.attract)
        self.history_threshold.append(self.threshold)
        self.history_performance.append(self.performance)

    def next_step(self):
        self.update_behaviour()
        self.perform_behaviour()
        self.update_individual_learning()
        self.save_data_behaviour()

