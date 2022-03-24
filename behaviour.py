class Behaviour():
    """ 
        Class for behaviours
        Properties: behaviour name, value, attract, cost
    """
    def __init__(self, behaviour_name, behaviour_type, init_value, init_attract, init_cost,behaviour_cap):
        self.behaviour_name = behaviour_name
        self.behaviour_type = behaviour_type
        self.value = init_value
        self.attract = init_attract
        self.cost = init_cost

        self.behaviour_cap = behaviour_cap

        self.history_value = [self.value]
        self.history_attract = [self.attract]
        self.history_cost= [self.cost]

        #print("init behaviour",self.behaviour_name, self.behaviour_type )

    def update_behaviour(self):
        value = self.attract - self.cost #add some biased transmission
        
        #print("value", value)
        if value < -self.behaviour_cap:#cap it at 5 for now 
            value = -self.behaviour_cap
            #print("less")
        elif value > self.behaviour_cap:
            value = self.behaviour_cap
            #print("more")
        
        self.value = value

    def save_data_behaviour(self):
        self.history_value.append(self.value)
        self.history_attract.append(self.attract)
        self.history_cost.append(self.cost)

    def next_step(self):
        self.update_behaviour()
        self.save_data_behaviour()

