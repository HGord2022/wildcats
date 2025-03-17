from sim import WildcatModel
from priors import priors
import random
import pandas as pd
from summary import summary_stats
import os
import pickle
import time

'''
This script is called by the "run_sim_round1.sh" HPC job script. It initiates a simulation based on 
parameters sampled randomly from the prior. 
'''

print("### Simulation starting ###")

priors_df = pd.DataFrame(columns=priors.keys())
rand = random.randint(1,999999)

samples = []
for key, prior in priors.items():
    samples.append(float(prior.rvs(1)))

prior_dict = dict(zip(priors.keys(),samples))
thetas = pd.DataFrame(prior_dict, index=['i', ])
array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

print(prior_dict)

# point values => recombination_rate=1.8e-8, mutation_rate= 0.86e-8

model = WildcatModel(seq_length=int(44648254), recombination_rate=prior_dict["recombination_rate"], mutation_rate=prior_dict["mutation_rate"])

data, time =  model.simulate(
        bottleneck_strength_domestic = prior_dict["bottleneck_strength_domestic"],
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
        n_samples=[6, 63, 22], # 6 domestic, 63 wild, 22 captive
        seed=rand)

# 'n_samples' must be changed to reflect the dataset

# calculate summary stats (saves file directly)
summary_stats(data)

# save the parameter set and time taken for simulation to pickle files to be merged
filename = "./output/thetas/theta%s.pickle" % array_id

with open(filename, 'wb') as handle:
    pickle.dump(thetas, handle, protocol=pickle.DEFAULT_PROTOCOL)

filename1 = "./output/times/time%s.pickle" % array_id

with open(filename1, 'wb') as handle:
    pickle.dump(time, handle, protocol=pickle.DEFAULT_PROTOCOL)


print("### Simulation finished ###")
