from datetime import date

from pandas.core.arrays.sparse import dtype
import numpy as np
from numpy.lib.mixins import _inplace_binary_method
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickFileWriter
import numpy.random as npr
from time import sleep, time
from matplotlib.widgets import Button

from rules import * 
from population import Population, State

from matplotlib import style
style.use('dark_background')
import matplotlib.animation as animation


class XYVis:
    def __init__(self, sim, dot_size=2, p=0.9):
        """
        Creates  and starts the animation of simulation.

        Args:
            sim (object): Instance if Sim
            dot_size (int, optional): Size of an agent in visualization . Defaults to 2.
        """
        self.dot_size = dot_size
        self.p = p
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6)) # ax is a list
        # self.fig.suptitle('Click to pause/unpause', fontsize=20)
        self.sim = sim
        self.anim = FuncAnimation(self.fig, self.update,
                                  init_func=self.setup, 
                                  interval=10)
        self.paused = False

        # Pause button
        pause_ax = self.fig.add_axes((0.2, 0.945, 0.6, 0.05))
        pause_button = Button(pause_ax, 'pause', color='black',hovercolor='0.2')
        pause_button.label.set_fontsize(18)
        pause_button.on_clicked(self.toggle_pause)

        plt.show()
    
    def toggle_pause(self, *args, **kwargs):
        """
        Run a callback when desired events are generated (pause/unpause animation).
        """
        if self.paused:
            self.anim.event_source.stop()
        else:
            self.anim.event_source.start()
        self.paused = not self.paused
    
    def data(self):
        """
        Returning  agents' coordinates and state.

        Returns:
            [object]: specifics(coordinates and state) about agents
        """
        return self.sim.pop.all["X"], \
               self.sim.pop.all["Y"], \
               self.sim.pop.all["State"].values 
               
    
    def setup(self):
        """
        Method is used to draw the simulation and graph.

        Returns:
            [object]: created scatter plot and graph(with specific states)
        """
        # Simulation
        x, y, c = self.data()
        self.scat = self.ax[0].scatter(x=x, y=y, c=c)
        self.ax[0].set_title('Simulation', fontsize=16)
        self.ax[0].set_xlabel('X')
        self.ax[0].set_ylabel('Y')

        # Hisory plot
        self.sus, = self.ax[1].plot([], [], c='white', lw=2, label='Susceptible')
        self.inf, = self.ax[1].plot([], [], c='red', lw=2, label='Infected')
        self.rec, = self.ax[1].plot([], [], c='green', lw=2, label='Recovered')
        self.dead, = self.ax[1].plot([], [], c='blue', lw=2, label='Dead')
        self.ax[1].legend( shadow=True)
        self.ax[1].set_title('Graph', fontsize=16)
        self.ax[1].set_xlabel('Time(days)')
        self.ax[1].set_ylabel('Number of people')

        return self.scat, self.sus, self.inf, self.rec, self.dead
    
    def SIRD(self):
        """
        Returns value of day and agents in every possible state from history of simulation to draw the graph.

        Returns:
            [object]: Informations about each day.
        """
        h = self.sim.history
        return h.index.values, h["Sus"].values, h["Inf"].values, h["Rec"].values, h["Dead"].values
    
    def update(self, _):
        """Updates pieces of the animation(coordinates of an agent with specific color).
        Method is also responsible for expanding the graph every iteration.

        Returns:
            [object]: updated animation, with prepared visualization of agents in simulation.
        """
        try:
            next(sim)
        except StopIteration:
            self.toggle_pause()
            print(f"R0 = {sim.R_zero(self.p)}")
        
        x, y, c = self.data()
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        coords = np.hstack([x, y])
        self.scat.set_offsets(coords)
        self.scat.set_sizes(np.full(len(coords), self.dot_size))
        self.scat.set_color(c)
        
        days, S, I, R, D = self.SIRD()
        
        self.sus.set_data(days, S)
        self.sus.set_color(State.SUS.value)
        
        self.inf.set_data(days, I)
        self.inf.set_color(State.INF.value)
        
        self.rec.set_data(days, R)
        self.rec.set_color(State.REC.value)
        
        self.dead.set_data(days, D)
        self.dead.set_color(State.DEAD.value)
        
        self.ax[1].set_xlim((0, days[-1] + 1))
        self.ax[1].set_ylim((0, self.sim.pop_size))
        
        return self.scat, self.sus, self.inf, self.rec, self.dead


class Sim:
    def __init__(self, sus, infected, rules):
        """
        Main simulation object, responsible for running the simulation.
        Use the start method to start the simulation.
        
        The simulation object can be iterated. Every iteration representing one "day"
        of the simulation. Each "day" every rule is applied to the population.

        
        pop_size: number of agents in the simulation
        day: counter of days in the simulation
        history: DataFrame with summarised  information from every day 

        Args:
            sus (int): number of susceptibles at the beginning of the simulation
            infected (int): number of infected at the beginning of the simulation
            rules (list[Rule]): list of rules that will be applied (in order) to the population in every iteration
        """
        self.rules = rules
        self.pop = Population(sus, infected)
        self.pop_size = sus + infected
        self.day = 0
        
        self.history = pd.DataFrame(columns=["Sus", "Inf", "Rec", "Dead", "Vaccinated"])
        self.history.index.name = "Day"
        
    def __stop_condition(self):
        return len(self.pop.infected) == 0 
        # return len(self.pop.infected) == 0 or len(self.pop.sus) == 0

    def __next__(self):
        """ 
        Execute new iteration (day) of the simulation by applying every rule.
        Create summary of a day and stores it in self.history.
        
        Raises:
            StopIteration: Raised when end day of simulation is reached.
        """
        if self.day <= self.end_day and not self.__stop_condition():
            start = time()
            
            self.history = self.history.append(pd.Series(name=self.day, dtype=object))
            
            for rule in self.rules:
                rule.apply(self.pop)
            
            self.history.loc[self.day, "Inf"] = len(self.pop.infected)
            self.history.loc[self.day, "Sus"] = len(self.pop.sus)
            self.history.loc[self.day, "Dead"] = len(self.pop.dead)
            self.history.loc[self.day, "Rec"] = len(self.pop.recovered)
            self.history.loc[self.day, "Quarantined"] = len(self.pop.quarantined)
            
            end = time()
            
            if self.debug:
                print(self.history.loc[self.day])
                print("t: {:.4f}".format(end - start))
                print("--------------------------")
                
            self.day += 1
        else:
            raise StopIteration
        
    def start(self, visual=None, vis_args=None, end_day=float('inf'), debug=False):
        """
        Method that launches the simulation.
        If visual is not None, a new object of type visual is created. 
        Otherwise the simulation runs without the visualization

        Args:
            visual (Class, optional): Class implementing visualization. The constractor must take at least one argument - simulation,  
                and must start the visualization. Defaults to None.
            vis_args (tuple, optional): Other parameters for vizualizations' constructor. Defaults to None.
            end_day (int, optional): Limit of simulation iteration. Defaults to float('inf') (infinity), meaning simulation runs forever.
            debug (bool, optional): Print daily summaries to terminal. Defaults to False.
        """

        self.end_day = end_day
        self.debug = debug
        
        self.__setup()
        
        if visual:
            self.vis = visual(self, *vis_args) if vis_args else visual(self)
        else:
            try:
                while True:
                    next(self)
            except StopIteration:
                pass
                
    def __setup(self): 
        """ 
        Runs setup for every rule.
        """
        for rule in self.rules:
            rule.setup(self, self.pop)

    def to_csv(self, path):
        """
        Saves history of simulation to csv

        Args:
            path (string): file path
        """
        self.history.to_csv(path,  sep='\t', encoding='utf-8')
        

    def R_zero(self, p):
        return self.day * self.history["AvgContacts"].mean() * p



        
rules = [ 
    XYInit(xy_bounds=(-50, 50), distribution = npr.uniform), # must be run before anything else
    XYMove(),
    # SetRateRadius(),
    # AdditionalProtection(maskOnProcentage = 0.1,  mask_rate = 0.99),
    # RandomizedTesting(daily_tests=30, recovery_rate=0.1, death_rate=0.03, dayt = 0),
    # Vaccination(vaccineProcentage=1, vaccination_perDay = 30, vaccine_rate = 0.90, dayTest = 6 ),
    # InfectPopulation(),
    InfectNoParameters(inf_radius = 1.5, inf_rate = 0.7),
    RecoverUniform(recovery_rate=0.1),
    DieUniform(death_rate=0.01)
]

sim = Sim( 1000, 100, rules)

sim.start(debug=True, visual=XYVis, vis_args=(1, 0.7),end_day=209)
sim.to_csv('out.csv')

