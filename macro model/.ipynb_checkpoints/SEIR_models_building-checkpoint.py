import pickle
import pandas as pd
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from Homogenenous_Trajectory_Network import HomogeneousTrajectoryNetwork
from Heterogeneous_Trajectory_Network import HeterogeneousTrajectoryNetwork
from Heterogeneous_SEIR import HeterogeneousSEIR




# PARAMETERS 1
infect_rate =  [3e-4, 15e-5] # second^-1
recover_prob = [1, 1]
exposed2infectious_prob = [1, 1]
q = [0, 0]

time_step = 1 # days
t_incubation = [1, 2] # days
t_infectious = [2, 4] # days
effective_duration =0.5*24*60*60 # second

RUNS = 5000

if __name__ = "__main__":

    # data 
    data_file = os.path.join(os.path.expanduser("~"), "FYP/data/synthetic_conference_attendance_data")
    with open(data_file, 'rb') as file:
        imported_dfs = pickle.load(file)

    unique_ids = pd.unique(pd.concat([imported_dfs[0]["id_1"], imported_dfs[0]["id_2"],imported_dfs[1]["id_1"], imported_dfs[1]["id_2"]]))

    
    # trajectory networks
    homogeneous_network = HomogeneousTrajectoryNetwork(list_of_dfs=imported_dfs)
    heterogeneous_network = HeterogeneousTrajectoryNetwork(list_of_dfs=imported_dfs)

    # SEIR models 
    SEIR_homo_model = HeterogeneousSEIR(trajectory_network=homogeneous_network, s_initial= s_intial, i_initial=i_inital, infect_rate=infect_rate, exposed2infectious_prob=exposed2infectious_prob, recover_prob=recover_prob, q=q, t_incubation=t_incubation, t_infectious=t_infectious, effective_duration=effective_duration, time_step=time_step)
    SEIR_hete_model = HeterogeneousSEIR(trajectory_network=heterogeneous_network, s_initial= s_intial, i_initial=i_inital, infect_rate=infect_rate, exposed2infectious_prob=exposed2infectious_prob, recover_prob=recover_prob, q=q, t_incubation=t_incubation, t_infectious=t_infectious, effective_duration=effective_duration, time_step=time_step)
