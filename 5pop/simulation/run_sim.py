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

model = WildcatModel(seq_length=int(45e6), recombination_rate=prior_dict["recombination_rate"], mutation_rate=prior_dict["mutation_rate"])

data, time =  model.simulate(
        captive_time=prior_dict["captive_time"],
        div_time=prior_dict["div_time"],
        div_time_dom=prior_dict["div_time_dom"],
        div_time_scot=prior_dict["div_time_scot"],
        mig_rate_captive=prior_dict["mig_rate_captive"],
        mig_rate_scot=prior_dict["mig_rate_scot"],
        mig_length_scot=prior_dict["mig_length_scot"],
        pop_size_captive=prior_dict["pop_size_captive"],
        pop_size_domestic_1=prior_dict["pop_size_domestic_1"],
        pop_size_lyb_1=prior_dict["pop_size_lyb_1"],
        pop_size_lyb_2=prior_dict["pop_size_lyb_2"],
        pop_size_scot_1=prior_dict["pop_size_scot_1"],
        pop_size_eu_1=prior_dict["pop_size_eu_1"],
        pop_size_eu_2=prior_dict["pop_size_eu_2"],
        n_samples=[6, 65, 22, 15, 4],
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
