class Behaviour():
    """ 
        Class for behaviours
        Properties: behaviour name, value, attract, threshold
    """
    def __init__(self, init_attract, init_threshold,carbon_emissions,phi,delta_t,save_data):
        
        self.attract = init_attract
        self.init_threshold = init_threshold #for the sake of being able to fix the carbon price
        self.threshold = init_threshold
        self.value = self.calc_behaviour()
        self.carbon_emissions = carbon_emissions
        self.delta_t = delta_t
        self.save_data = save_data
        self.phi = phi

        if self.save_data:
            self.history_value = [self.value]
            self.history_attract = [self.attract]
            self.history_threshold= [self.threshold]

    def calc_behaviour(self):
        value = self.attract - self.threshold 
        return value

    def update_attract(self,social_learing):
        #print("attract cahnge: ",self.attract,social_learing,self.delta_t*(self.phi*(self.attract - social_learing)))
        self.attract += self.delta_t*(social_learing)#self.delta_t*(self.phi*(self.attract - social_learing))
        #quit()
    
    def save_data_behaviour(self):
        self.history_value.append(self.value)
        self.history_attract.append(self.attract)
        self.history_threshold.append(self.threshold)

    def next_step(self):
        self.value = self.calc_behaviour()
        if self.save_data :
            self.save_data_behaviour()

