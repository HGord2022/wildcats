from sim import WildcatModel
from priors import priors
import random
import pandas as pd
import numpy as np
from summary import summary_stats
import os
import pickle
import time

'''
This script is called by the "run_sim_sequential.sh" HPC job script. It initiates a simulation based on 
parameters sampled from the previous round's posterior distribution (params_rn.csv).
'''

print("### Simulation starting ###")
rand = random.randint(1,999999)
array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

# load params sampled from posterior and select new row
df = pd.read_csv("params_r2.csv")
params = list(df.iloc[array_id])

prior_dict = dict(zip(priors.keys(),params))
thetas = pd.DataFrame(prior_dict, index=['i', ])
filename = "./output/thetas/theta%s.pickle" % array_id

print(prior_dict)

model = WildcatModel(seq_length=int(44648254), recombination_rate=prior_dict["recombination_rate"], mutation_rate=prior_dict["mutation_rate"])

data, time =  model.simulate(
        bottleneck_strength_domestic = prior_dict["bottleneck_strength_domestic"]
        bottleneck_strength_wild = prior_dict["bottleneck_strength_wild"],
        bottleneck_time_domestic = prior_dict["bottleneck_time_domestic"],
        bottleneck_time_wild = prior_dict["bottleneck_time_wild"],
        captive_time = prior_dict["captive_time"],
        div_time = prior_dict["div_time"],
        mig_length_post_split = prior_dict["mig_length_post_split"],
        mig_rate_post_split = prior_dict["mig_rate_post_split"],
        mig_length_wild = prior_dict["mig_length_wild"],
        mig_rate_wild = prior_dict["mig_rate_wild"],
        mig_rate_captive = prior_dict["mig_rate_captive"],
        pop_size_captive = prior_dict["pop_size_captive"],
        pop_size_domestic_1 = prior_dict["pop_size_domestic_1"],
        pop_size_domestic_2 = prior_dict["pop_size_domestic_2"],
        pop_size_wild_1 = prior_dict["pop_size_wild_1"],
        pop_size_wild_2 = prior_dict["pop_size_wild_2"],
        n_samples=[6, 65, 22], # 6 domestic, 65 wild, 22 captive
        seed=rand)

summary_stats(data)

# save parameter set and time taken for simulation

with open(filename, 'wb') as handle:
    pickle.dump(thetas, handle, protocol=pickle.DEFAULT_PROTOCOL)

filename1 = "./output/times/time%s.pickle" % array_id

with open(filename1, 'wb') as handle:
    pickle.dump(time, handle, protocol=pickle.DEFAULT_PROTOCOL)


print("### Simulation finished ###")
