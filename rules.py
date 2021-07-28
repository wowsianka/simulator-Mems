import abc
import numpy.random as npr
import numpy as np
from population import State
from scipy.spatial import KDTree
from sklearn.datasets import make_blobs

class Rule(abc.ABC):
    """
    Abstract interface for a Rules.
    """
    @abc.abstractmethod
    def setup(self, sim, pop):
        """Initialize everything that's needed for applying the Rule.

        Args:
            sim (Sim)
            pop (Population)
        """
        self.sim = sim
    
    @abc.abstractmethod
    def apply(self, pop):
        """Implementation of the behaviour of the Rule.

        Args:
            pop (Population)
        """
        pass
    

class XYNotInitialized(Exception):
    def __init__(self):
        """
        Exception takes an optional string argument message that gets printed with exception.  Value is passed to XYNotInitialized
        object, which constructs a special message and passes it its parent class, Exception, via super().__init__(). 
        The custom message string, along with the value for context, gets printed along with our error traceback.

        """
        super().__init__("Columns X, Y are not initialized. Rule XYInit has to be run before any other rule containing 'XY'")


class XYInit(Rule):
    def __init__(self, xy_bounds, distribution=npr.uniform):
        """
        Creates a rule that initializes all the fields needed for a simulation with agents moving on a X,Y plane.

        Args:
            xy_bounds (tuple): 
            distribution ( optional): Samples are uniformly distributed. Defaults to npr.uniform.
        """
        super().__init__()
        self.low, self.high = xy_bounds
        self.distribution = distribution
    
    def setup(self, sim, pop):
        """
        Adds X,Y columns to the Population with random coordinates from the given distribution. 

        Args:
            sim (Sim)
            pop (Population)
        """
        def set_axis(axis):
            pop.all[axis] = self.distribution(self.low, self.high, self.sim.pop_size)
            #  Case if distrbution is normal:           
            #  pop.all[axis] = self.distribution(0, 50/3, self.sim.pop_size)
             
        sim.xy_bounds = (self.low, self.high)
        self.sim = sim
        set_axis("X")
        set_axis("Y") 
        # # Clusters of agents:
        # coords, _ =  make_blobs(n_samples=len(pop.all), centers=30, n_features=2, random_state=0, center_box=(-50,50))
        # pop.all["X"], pop.all["Y"] = coords[:,0],coords[:,1]
        
        
    
    def apply(self, pop):
        pass
    

class XYHelper:
    @staticmethod
    def coords(df):
        """Returns coordinates X, Y of agents as a numpy array of shape (2, population size)

        Args:
            df (DataFrame): Contains coordinates X, Y of agents.

        Returns:
            [numpy array]: Stack coordinates X,Y arrays in sequence horizontally 
        """
        x = np.expand_dims(df["X"], axis=1)
        y = np.expand_dims(df["Y"], axis=1)
        return np.hstack([x, y])
    
    @staticmethod
    def check_if_initialized(sim, pop):
        """Checks if InitXY has been applied to the Sim.

        Args:
            sim (Sim)
            pop (Population)

        Raises:
            XYNotInitialized: throw expection if proper fields have not been initialized.
        """
        cols = pop.all.columns
        if 'X' not in cols or 'Y' not in cols:
            raise XYNotInitialized()
        
        try:
            _ = sim.xy_bounds
        except AttributeError:
            raise XYNotInitialized()


class XYMove(Rule):
    def __init__(self, move_distrib=npr.uniform, distrib_args=(-1, 1)):
        """
        Implements moving of agents in the sim.

        Args:
            low (float): lower x, y boundry
            high (float): upper x, y boundry
            move_distrib (random number generation, optional): Distribution of the sizes of steps
                that each agent takes when moving. Defaults to npr.uniform.
    
        """
        super().__init__()
        self.move_distrib = move_distrib
        self.distrib_args = distrib_args
    
    def setup(self, sim, pop):
        self.sim = sim
        XYHelper.check_if_initialized(sim, pop)
        self.low, self.high = sim.xy_bounds
    
    def apply(self, pop):
        """
        Changes the position of agents by a random offset from move_distrib.

        Args:
            pop (Population)
        """
        def move_axis(axis):
            step = self.move_distrib(*self.distrib_args, len(pop.all))
            
            pop.all[axis] += step
           
            # adjust bounds 
            pop.all.loc[pop.all[axis] > self.high, axis] = self.high
            pop.all.loc[pop.all[axis] < self.low, axis] = self.low
        
        move_axis("X")
        move_axis("Y")

class SetRateRadius(Rule):  
    def __init__(self, inf_radius=1.5, inf_rate=0.7):
        self.inf_radius = inf_radius
        self.inf_rate = inf_rate
    def setup(self, sim, pop):
        self.sim = sim
        pop.all["Inf_radius"] = self.inf_radius
        pop.all["Inf_rate"] = self.inf_rate
    def apply(self, pop):
        pass
            

class AdditionalProtection(Rule):   
    def __init__(self, inf_radius=1.5, inf_rate=0.7, 
    maskOnProcentage = 0.1,  mask_rate = 0.1):
        """
        Implements masks in the sim.

        Args:
            inf_radius (float, optional): Defaults to 1.5.
            inf_rate (float, optional): Defaults to 0.7.
            maskOnProcentage (int, optional): Defaults to 0.1
            mask_rate (int, optional):  Defaults to 0.
        """    
        super().__init__()

        self.inf_radius = inf_radius
        self.inf_rate = inf_rate

        self.maskOnProcentage = maskOnProcentage
        self.mask_rate = mask_rate

    def setup(self, sim, pop):
        """
        Applying masks on given percentage of population and changing their infectious rate.

        Args:
            sim (Sim)
            pop (Population)
        """        
        self.sim = sim
        # pop.all["Inf_radius"] = self.inf_radius
        # pop.all["Inf_rate"] = self.inf_rate
        pop.all["Mask"] =False

        def set_rate(procentage, rate):
            pop.all.loc[0:procentage *len(pop.all),"Inf_rate"] = pop.all.loc[0:procentage *len(pop.all),"Inf_rate"] * (1 - rate)
            pop.all.loc[0:procentage *len(pop.all),"Mask"] = True

        set_rate(self.maskOnProcentage, self.mask_rate)
        

    def apply(self, pop):
        self.sim.history.loc[self.sim.day, "Masks"] = len(pop.all[pop.all["Mask"]])
        # pass

class Vaccination(Rule):
    def __init__(self, inf_radius=1.5, inf_rate=0.7,  vaccineProcentage = 0.1, vaccination_perDay =  1, vaccine_rate = 0.5, dayTest = 0):
        """
        Rule that initializes vaccination of population, which changes values of infectious rate. Vaccination starts when sim.day meets dayTest.

        Args:
            inf_radius (float, optional): Defaults to 1.5.
            inf_rate (float, optional):  Defaults to 0.7.
            vaccineProcentage (int, optional):  Defaults to 0.1.
            vaccination_perDay (int, optional):  Defaults to 1.
            vaccine_rate (int, optional):  Defaults to 0.5.
            dayTest (int, optional): Defaults to 0.
        """        
        super().__init__()

        self.inf_radius = inf_radius
        self.inf_rate = inf_rate
        self.vaccineProcentage = vaccineProcentage
        self.vaccine_perDay = vaccination_perDay
        self.vaccine_rate = vaccine_rate
        self.dayTest = dayTest

    def setup(self, sim, pop):
        if "Inf_rate" not in pop.all:
            pop.all["Inf_radius"] = self.inf_radius
            pop.all["Inf_rate"] = self.inf_rate

        self.sim = sim
        pop.all["Vaccinated"] =False
        self.priorSus = len(pop.sus)

    def apply(self, pop):
        """
        Every iteration (dayTest is smaller than sim.day) vaccine_perDay agents with status Susceptible are vaccinated , until procentage of vaccinated people is met. 

        Args:
            pop (Population)
        """        
        if(self.dayTest <= self.sim.day):
            counter = 0
            # for index in range(len(pop.all)):
            for index, row in pop.all.iterrows():
                vaccinated_people = sum(pop.all['Vaccinated'])
                if(vaccinated_people<self.vaccineProcentage*self.priorSus and counter< self.vaccine_perDay):
                    if(pop.all.loc[index, "State"] == State.SUS.value):
                        if(pop.all.loc[index, "Vaccinated"] == False):
                            pop.all.loc[index,"Vaccinated"]  = True
                            pop.all.loc[index,"Inf_rate"]  = pop.all.loc[index,"Inf_rate"]*(1-self.vaccine_rate) 
                            counter += 1
                else:
                    pass
        self.sim.history.loc[self.sim.day, "Vaccinated"] = len(pop.all[pop.all["Vaccinated"]])
            

class InfectPopulation(Rule):

    def __init__(self):
        super().__init__()       

    def setup(self, sim, pop):      
        self.sim = sim
        XYHelper.check_if_initialized(sim, pop)
        self.low, self.high = sim.xy_bounds
        
        inf_idx = pop.all["State"] == State.INF.value
        pop.all.loc[inf_idx, "Inf Day"] = sim.day
        pop.all.loc[~inf_idx, "Inf Day"] = np.nan

    def apply(self, pop):
        """[summary]
        Rule is responsible for infection of agents, who are located within given radius and infects them if the probability of infection is lower.
        Args:
            pop (Population)
        """        
        inf = pop.infected
        sus = pop.sus

        if len(inf)==0 or len(sus)==0: return
        inf_tree = XYHelper.coords(inf)
        sus_tree = XYHelper.coords(sus)

        def arg_within_radius(inf,susceptible,r):
            dist = np.sqrt(((inf-susceptible)**2).sum(axis=1))
            return np.argwhere(dist<r).ravel()

        for index in range(len(inf)):
            infected = inf_tree[index]
            radius = inf.iloc[index]["Inf_radius"]
            rate = inf.iloc[index]["Inf_rate"]
            infect = arg_within_radius(infected,sus_tree,radius)
            real_index = sus.iloc[infect].index
            to_infect = pop.all.loc[real_index, "Inf_rate"] > npr.random(len(real_index))
            to_infect = real_index[to_infect]
            pop.all.loc[to_infect, "State"] = State.INF.value
            pop.all.loc[to_infect, "Inf Day"] = self.sim.day
            
            this = inf.iloc[index].name
            pop.all.loc[this , "Contacts"] = len(infect)

        self.sim.history.loc[self.sim.day, "AvgContacts"] = pop.all["Contacts"].mean()
        

class InfectNoParameters(Rule):
    def __init__(self, inf_radius=1.5, inf_rate=0.7):
        """
        Infection of agents is based on  KDTree, which finds all the agents within given radius and infects them if the porbability of infection is lower.
        Args:
            inf_radius (float, optional): Agents are infected in inf_radius, radius is used in KDTree. Defaults to 1.5.
            inf_rate (int, optional): Probability of infection. Defaults to 0.7.
        """
        super().__init__()
        self.inf_radius = inf_radius
        self.inf_rate = inf_rate

    def setup(self, sim, pop):
        self.sim = sim
        XYHelper.check_if_initialized(sim, pop)
        self.low, self.high = sim.xy_bounds
        
        inf_idx = pop.all["State"] == State.INF.value
        pop.all.loc[inf_idx, "Inf Day"] = sim.day
        pop.all.loc[~inf_idx, "Inf Day"] = np.nan

    def apply(self, pop):
        """
        Agents are being infected using  KDTree.
        Args:
            pop (Population)
        """
        inf = pop.infected
        sus = pop.sus
        
        if len(inf)==0 or len(sus)==0: return

        inf_tree = KDTree(XYHelper.coords(inf))
        sus_tree = KDTree(XYHelper.coords(sus))

        indexes = inf_tree.query_ball_tree(sus_tree, r=self.inf_radius)
        
        for i in range(len(indexes)):
            this = inf.iloc[i].name
            pop.all.loc[this , "Contacts"] = len(indexes[i])

            for j in indexes[i]:
                if npr.random() < self.inf_rate:
                    real_idx = int(sus.iloc[j].name)
                    pop.all.loc[real_idx, "State"] = State.INF.value
                    pop.all.loc[real_idx, "Inf Day"] = self.sim.day              

        self.sim.history.loc[self.sim.day, "AvgContacts"] = pop.all["Contacts"].mean()             

class RecoverUniform(Rule):
    def __init__(self, recovery_rate, infectionPeriod = 0):
        """
        Recover infected using a uniform distribution. 
        Recover if the value from the distribution  is lower than recovery rate.

        Args:
            recovery_rate (float):  probability of recovery
        """
        super().__init__()
        self.recovery_rate = recovery_rate
        self.infection_period = infectionPeriod
    
    def setup(self, sim, pop):
        """
        #TODO
        Before launching simulation pop.all["Rec Day"] is not existing.

        Args:
            sim (Sim)
            pop (Population)
        """
        self.sim = sim
        pop.all["Rec Day"] = np.nan
    
    def apply(self, pop):
        """
        Method that recover infected agents when  the value from the distribution  
        is lower than recovery rate. Also remebers day when particular infected agents recover.
        Args:
            pop (Population)
        """
        inf = pop.infected[self.sim.day - pop.infected["Inf Day"] > self.infection_period ] 
        real_idx = inf[npr.uniform(0, 1, size=len(inf)) < self.recovery_rate].index
        pop.all.loc[real_idx, "State"] = State.REC.value 
        pop.all.loc[real_idx, "Rec Day"] = self.sim.day


class DieUniform(Rule):
    def __init__(self, death_rate, infectionPeriod =0):
        """
        Agents die if the value from the uniform distribution  is lower than death rate.

        Args:
            death_rate (float): probability of death
        """
        super().__init__()
        self.death_rate = death_rate
        self.infection_period = infectionPeriod

    def setup(self, sim, pop):
        """
        Before launching simulation pop.all["Death Day"] is not existing.

        Args:
            sim (Sim)
            pop (Population)
        """
        self.sim = sim
        pop.all["Death Day"] = np.nan
        
    def apply(self, pop):
        """
        Method that turns infected agents into dead when  the value from the distribution  
        is lower than death rate. Also remebers day when particular infected agents died.
        Args:
            pop (Population)
        """
        inf = pop.infected[self.sim.day - pop.infected["Inf Day"] > self.infection_period ]
        real_idx = inf[npr.uniform(0, 1, size=len(inf)) < self.death_rate].index
        pop.all.loc[real_idx, "State"] = State.DEAD.value
        pop.all.loc[real_idx, "Death Day"] = self.sim.day
        pop.remove(real_idx)

        
class RandomizedTesting(Rule):
    def __init__(self, daily_tests, death_rate, recovery_rate, dayt = 0):
        self.daily_tests = daily_tests
        self.death_rate = death_rate
        self.recovery_rate = recovery_rate
        self.dayTEST = dayt
    
    def setup(self, sim, pop):
        self.sim = sim        
    
    def apply(self, pop):
        """Generating random samples of agents, if status of an agent is infected he change to status quarantine. Then using a uniform distribution, agents 
        recover if the value from the distribution  is lower than recovery rate and die if the value from the uniform distribution  is lower than death rate

        Args:
            pop ([type]): [description]
        """        
        if(self.dayTEST <= self.sim.day):
            tested = npr.choice(pop.all.index, size=(min(self.daily_tests, len(pop.all))))
            to_quarantine = tested[(pop.all.loc[tested, "State"] == State.INF.value).values]
            pop.all.loc[to_quarantine, "State"] = State.QUARANTINE.value
            
            quarantined = pop.all[pop.all["State"] == State.QUARANTINE.value]
            self.sim.history.loc[self.sim.day, "Quarantined"] = len(quarantined)
            
            # quarantine recovery
            real_idx = quarantined[npr.uniform(0, 1, size=len(quarantined)) < self.recovery_rate].index
            pop.all.loc[real_idx, "State"] = State.REC.value 
            pop.all.loc[real_idx, "Rec Day"] = self.sim.day
            
            # quarantine death
            real_idx = quarantined[npr.uniform(0, 1, size=len(quarantined)) < self.death_rate].index
            pop.all.loc[real_idx, "State"] = State.DEAD.value
            pop.all.loc[real_idx, "Death Day"] = self.sim.day
            pop.remove(real_idx)
        