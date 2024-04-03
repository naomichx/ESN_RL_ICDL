# Exploring the role of structure with ESN-RL


Code for the ICDL paper : [Exploring the role of structure in
a time constrained decision task](
https://doi.org/10.48550/arXiv.2401.10849
)


## Repository structure

The repository is organized as follows:

`json_files/`: This directory contains JSON files with task and model parameters.\
`task.py`: Python file that contains the time-constrained decision task implementation.\
`model.py`: Python file that contains all the ESN and RL models used in the project.
`experiment.py`: Python file that allows you to implement the whole experiment and save the data.\
`main.py`: Main Python file to run the whole simulation.\
`utils.py`: Python file that contains various utility functions and tools.\
`results_analysis.py`: Python file that allows you to analyze all the simulations and plot the performances.

In `main.py` choose the type of the model you want to run `model_type`, if you want to have a testing phase: `testing=True`, if you want to show the plots at the end of a simulation: show_plots=True, if you want to save save=True, set the number of simulation you want to run on the same model: n_seed. 


## Running the Simulation
To run the simulation, modify the main.py file according to your preferences:

Choose the type of model you want to run by setting the `model_type` variable.\
Set `testing=True` if you want to include a testing phase.\
Set `show_plots=True` if you want to display the plots at the end of a simulation.\
Set `save=True` if you want to save the simulation data.\
Set the number of simulations you want to run on the same model by modifying the `n_seed` variable.\
After setting the desired parameters, simply run main.py to start the simulation.\

## Analyzing the Results
To analyze the simulation results and plot the performances, use the `results_analysis.py` file. Make sure you have saved the simulation data before running the analysis.

For more details on the project, please refer to [the ICDL paper](https://doi.org/10.48550/arXiv.2401.10849).