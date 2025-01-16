from sim_map import WildcatModel
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

params = [1.50174731e+04, 1.00140416e+04, 3.28731094e+03, 3.30199424e+03,
       1.59282276e+01, 6.02463252e+04, 4.34017358e+03, 1.31742661e+01,
       8.06616504e-02, 1.96057230e-02, 8.07042262e-02, 6.05481604e+03,
       8.13188096e+03, 1.11097647e+02, 6.35044577e+03, 9.88676558e+03]


prior_dict = dict(zip(priors.keys(),params))
print(prior_dict)

model = WildcatModel(seq_length=int(44648254), recombination_rate=1.8e-8, mutation_rate= 0.86e-8)

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
        n_samples=[6, 63, 21], # 6 domestic, 63 wild, 21 captive
        seed=rand)


print("### Simulation finished ###")
