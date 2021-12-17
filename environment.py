# @author Dylan Goetting
class Environment: 
    """
    Environemt object, representing a simulated market. Characterized by the different demand distributions 
    for each day of the week, the unit profit, unit production cost, the number of weeks each episode will last,
    and the maximum possible demand.
    """
    
    def __init__(self, distributions, profit, cost, numWeeks, maxDemand):
        
        if not len(distributions) == 7:
            raise ("there must be 7 distributions")

        self.distributions = distributions
        self.day = 0
        self.unit_profit = profit
        self.unit_cost = cost 
        self.numWeeks = numWeeks
        self.maxDemand = maxDemand

    def step(self, action):
        """
        Perform one internal step of the environment, given the agent has chosen ACTION
        """
        # Sample from a demand distribution, make sure both the demand and action
        # are integers within the acceptable range
        demand = self.distributions[self.day % 7].sample()
        action = int(action)
        demand = int(demand)

        if action > self.maxDemand:
            action = self.maxDemand
        
        if action < 0:
            action = 0

        if demand > self.maxDemand:
            demand = self.maxDemand

        if demand < 0:
            demand = 0

        # Calculate the net profit the agent generated for this timestep
        sold = min(demand, action)
        unsold = action - sold
        profit = sold*self.unit_profit - unsold*self.unit_cost
        
        done = (self.day // 7 >= self.numWeeks)
        self.day += 1

        return profit, self.observe(), done

    def reset(self):
        self.day = 0

    def observe(self):
        return self.day%7

    def getMaxDemand(self):
        return self.maxDemand

