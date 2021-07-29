# simulator-Mems
Epidemic simulator based on multiagent system
Simulation takes the name "Mems". Mems is a library for quick and easy prototyping of simple epidemiological simulations. The whole program is based 
on the idea of rules. The main idea of the project is to create such a simulation, which is not necessarily the fastest and is executed on millions of agents,
but it is supposed to allow rapid prototyping of different behaviors in such a simulation. That is, in addition to designing the simplest simulation using 
the susceptible-infected-recovered/dead (SIR) model, we can extend it to include other behaviors that are less standard. The simulation allows us to design 
an unlimited number of rules by adding a single class implementing the standard Rules interface. An agent-based epidemic simulator gives great flexibility to the user. 
