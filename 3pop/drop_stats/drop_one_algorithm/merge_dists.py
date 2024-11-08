import pandas as pd
import numpy as np
import pyarrow
import os

df_list = []
missing_files = []
num_files = 101

for i in range(0, num_files):
    filename = "./hdr/hdr{}.pickle".format(i)
    try:

        df = pd.read_pickle(filename)
        df = df.reset_index(drop=True)
        df_list.append(df)
        os.remove(filename)
    except pyarrow.lib.ArrowIOError:
        missing_files.append(i)

df_list = pd.concat(df_list, axis=0).reset_index(drop=True)
hdr = pd.DataFrame(df_list)

hdr.to_csv("./hdr.csv", index=False)