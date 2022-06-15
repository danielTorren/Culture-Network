import numpy as np

class Behaviour():
    """ 
        Class for behaviours
        Properties: behaviour name, value, attract, threshold
    """
    def __init__(self, init_attract, init_threshold,carbon_emissions,attract_individual_learning,psi,phi,learning_error_scale):
        
        self.attract = init_attract
        self.init_threshold = init_threshold #for the sake of being able to fix the carbon price
        self.threshold = init_threshold
        self.value = self.calc_behaviour()
        self.carbon_emissions = carbon_emissions
        self.attract_individual_learning = attract_individual_learning
        self.performance = self.calc_perform()# AT SOME POINT THIS SHOUDL BE SET RADNOMLY

        #individual learning
        self.psi = psi
        self.individual_learning = self.calc_individual_learning()# AT SOME POINT THIS SHOUDL BE SET RADNOMLY

        self.phi = phi
        self.learning_error_scale = learning_error_scale
        self.gamma = self.calc_gamma()

        self.history_value = [self.value]
        self.history_attract = [self.attract]
        self.history_threshold= [self.threshold]
        self.history_performance = [self.performance]

    def calc_behaviour(self):
        value = self.attract - self.threshold 
        return value

    def calc_perform(self):
        """
        #TRYING TO USE LOGIT MODEL
        prob_performance = 1/(1 + np.exp(-self.value))
        self.performance = np.random.binomial(1, prob_performance)
        """
        if self.value > 0:
            return 1
        else:
            return 0

    def calc_gamma(self):
        #print(np.random.normal(loc = 1, scale = self.learning_error_scale))
        return np.random.normal(loc = 1, scale = self.learning_error_scale)

    def update_attract(self,social_component_behaviours,information_provision,delta_t):#,attract_cultural_group
            
            #social_learing = self.gamma*self.phi*(social_component_behaviours - self.attract) 
            social_learing = social_component_behaviours
            #individual_learning = self.individual_learning 
            #conformity_bias =  (attract_cultural_group - self.attract) 
            information_provision = information_provision 

            #total = social_learing  + individual_learning #+ conformity_bias + information_provision
            #print("PROPORTIONS (S,IL,CG,IP):", social_learing, individual_learning)#,conformity_bias/total,information_provision/total

            #print("attract components!",social_learing,individual_learning,conformity_bias,information_provision)
                #social_learing + individual_learning + 
            self.attract += delta_t*(social_learing + information_provision) #+ individual_learning 

    def update_threshold(self, carbon_price):
        if self.init_threshold < carbon_price*self.carbon_emissions:
            self.threshold = 0 
        else:
            self.threshold = self.init_threshold - carbon_price*self.carbon_emissions

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


    def save_data_behaviour(self):
        self.history_value.append(self.value)
        self.history_attract.append(self.attract)
        self.history_threshold.append(self.threshold)
        self.history_performance.append(self.performance)

    def next_step(self):
        self.value = self.calc_behaviour()
        self.performance = self.calc_perform()
        self.individual_learning = self.calc_individual_learning()
        self.calc_gamma()
        self.save_data_behaviour()

