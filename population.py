from enum import Enum
import numpy as np
import pandas as pd

class State(Enum):
    """
    Enum represents all of the states of an agent,
    with values being colors.
    """
    SUS = "white"
    INF = "red"
    REC = "green"
    DEAD = "blue"
    QUARANTINE = "coral"
    
    @classmethod
    def from_value(cls, value):
        """Given a color returns state.
        Args:
            value (string): color
        Returns:
            [State]: state which value is color
        """
        return {
            "white": cls.SUS,
            "red": cls.INF,
            "green": cls.REC,
            "blue": cls.DEAD,
            "yellow": cls.QUARANTINE
        }[value]


class Population(object):
    def __init__(self, sus, infected):
        """
        A wrapper object around a DataFrame containing all agents.
        Initializes a given number of susceptibles and infected agents.
        all: DataFrame containg all agents except with State.DEAD
        removed: DataFrame containg all agents with State.DEAD
        Args:
            sus (int): number of susceptibles at the beginning of the simulation
            infected (int): number of infected at the beginning of the simulation
        """
        states = np.repeat([State.INF.value, State.SUS.value],
                           [infected, sus])
        self.all = pd.DataFrame(states, columns=["State"])
        self.removed = pd.DataFrame()

    def __with_state(self, df, state):
        """
        Helper method returning all the agents with a given State.
        Args:
            df (DataFrame)
            state (State)
        Returns:
            (DataFrame)
        """
        return df[df["State"] == state.value]
    
    def remove(self, index):
        """
        Method tha adds removed agents to removed DataFrame
        and drops specified row from DataFrame with all agents.
        Args:
            index (int): number of row
        """
        rows = self.all.loc[index, :]
        self.removed = self.removed.append(rows, ignore_index=True)
        self.all.drop(rows.index, inplace=True)
    
    @property
    def infected(self):
        """
        DataFrame with infected agents
        """
        return self.__with_state(self.all, State.INF)
   
    @property
    def sus(self):
        """
        DataFrame with susceptible agents
        """
        return self.__with_state(self.all, State.SUS)
    
    @property
    def dead(self):
        """
        DataFrame with dead agents
        """
        return self.__with_state(self.removed, State.DEAD)
    
    @property
    def recovered(self):
        """
        DataFrame with recovered agents
        """
        return self.__with_state(self.all, State.REC)

    @property
    def quarantined(self):
        """
        DataFrame with quarantined agents
        """
        return self.__with_state(self.all, State.QUARANTINE)