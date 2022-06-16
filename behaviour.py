class Behaviour():
    """ 
        Class for behaviours
        Properties: behaviour name, value, attract, threshold
    """
    def __init__(self, init_attract, init_threshold,carbon_emissions,delta_t):
        
        self.attract = init_attract
        self.init_threshold = init_threshold #for the sake of being able to fix the carbon price
        self.threshold = init_threshold
        self.value = self.calc_behaviour()
        self.carbon_emissions = carbon_emissions
        self.delta_t = delta_t

    def calc_behaviour(self):
        value = self.attract - self.threshold 
        return value

    def update_attract(self,social_learing):
        self.attract += self.delta_t*(social_learing) 

    def next_step(self):
        self.value = self.calc_behaviour()

