from flowjax.train.data_fit import fit_to_data
from flowjax.train.losses import ContrastiveLoss
from flowjax.flows import masked_autoregressive_flow as MaskedAutoregressiveFlow
from flowjax.distributions import Normal
import flowjax.bijections as bij
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from jax import vmap
import pandas as pd
import json
import pickle
import csv
import os


# flatten dictionary to data frame function:


def flatten_dict(d, sep='_'):
    """
    Recursively flattens a nested dictionary, concatenating the outer and inner keys.

    Arguments
    -----------
    stats_dict: A nested dictionary of statistics
    sep: seperator for keys

    Returns
    ------------
    dict
    """

    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key + sep + subkey, subvalue
            else:
                yield key, value

    return dict(items())

# main:

#observed data
filename = "./observed_stats.csv"
x_o = pd.read_csv(filename)
print("observed shape", np.shape(x_o))
#simulated data
filename2 = "./summary_stats_r1_15k.csv"
x = pd.read_csv(filename2)
print("observed shape", np.shape(x))

array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])-1

print("This is array # ", array_id)

to_drop = [#correlation >0.99
    "y1_domestic",
    "y1_scot",
    "y1_captive",
    "y1_eu",
    "y1_lyb",
    "y3_domestic_captive_lyb",
    "divergence_domestic_eu",
    "y3_domestic_scot_lyb",
    "y3_domestic_eu_lyb",
    "divergence_captive_lyb",
    "segregating_sites_lyb",
    "segregating_sites_eu",
    "divergence_scot_lyb",
    "f4_domestic_scot_captive_lyb",
    "f4_domestic_captive_eu_lyb",
    "f4_domestic_scot_eu_lyb",
    "diversity_domestic",
    "y3_domestic_scot_captive",
    "y3_scot_captive_eu",
    "f2_domestic_eu",
    "y2_domestic_lyb",
    "y3_scot_captive_lyb",
    "divergence_captive_eu",
    "pc1_dist_domestic_eu",
    # outlier noise model method r1 (>1)
    'tajimas_d_eu',
    'relatedness_domestic_lyb',
    'pc1_iqr_domestic',
    'pc1_iqr_eu',
    'pc1_iqr_lyb',
    'pc2_iqr_domestic',
    'pc2_iqr_lyb',
    # outlier noise model method r2 (>0.8)
    'relatedness_captive_lyb',
    'pc1_median_eu',
    'pc2_dist_domestic_lyb']
    # drop_one algorithm

combined_x = pd.concat([x, x_o], ignore_index=True)
combined_x = combined_x.drop(columns=to_drop)
statnames = combined_x.columns
combined_x = combined_x.drop(columns=statnames[array_id])

combined_x = combined_x.to_numpy(dtype=np.float32)
np.shape(combined_x)


#normalise data
x_scaler = StandardScaler()
print("combined_x shape", np.shape(combined_x))
combined_x_t = x_scaler.fit_transform(combined_x)
x_t = np.float32(combined_x_t[0:10000])
x_t_test = np.float32(combined_x_t[10000:15000])
x_o_t = np.float32(combined_x_t[15000])
x_o_t = np.reshape(x_o_t, (1,-1))
print("x_t shape", np.shape(x_t))
print("x_o_t shape", np.shape(x_o_t))
print("x_t_test shape", np.shape(x_t_test))

key, subkey = jr.split(jr.PRNGKey(2))
#define prior
n_summaries = len(statnames)-1
unbounded_prior = Normal(jnp.zeros((n_summaries,)))

flow = MaskedAutoregressiveFlow(
    subkey,
    base_dist=Normal(jnp.zeros((n_summaries,))),
)

import optax
optimizer = optax.chain(
        optax.clip_by_global_norm(1),
        optax.adam(5e-5),
    )

print("fitting flow")
fitted_flow, losses_r = fit_to_data(
    key=subkey,
    dist=flow,
    x=x_t,
    optimizer = optimizer,
    max_epochs=1500,
    show_progress=True,
    max_patience=20,
    batch_size=50
)

for k, v in losses_r.items():
    plt.plot(v, label=k)
plt.legend()
plt.savefig("./figs/losses.png")

# log prob of observation
print("calculating log prob of observed")
posterior = fitted_flow
obs = posterior.log_prob(x_o_t)

#sample from the posterior and find log probs
print("sampling and calculating log prob of 5000 thetas from posterior")
samples = x_t_test
log_probs = []
for i in range(len(samples)):
    log_probs.append(posterior.log_prob(samples[i]))

# calculate percentile
print("calculating percentile")
num = int(0)
for prob in log_probs:
    if obs < prob:
        num += 1

percentile = num/5000

print("Percentile for ", statnames[array_id], " is ", percentile)

hdr_dict = {}

hdr_dict["statname"] = statnames[array_id]
hdr_dict["percentile"] = percentile

final_array = pd.DataFrame(flatten_dict(hdr_dict), index=['i', ])

filename = "./hdr/hdr%s.pickle" % array_id

with open(filename, 'wb') as handle:
    pickle.dump(final_array, handle, protocol=pickle.DEFAULT_PROTOCOL)