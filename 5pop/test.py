import pandas as pd
df_list = []
df = pd.read_pickle("sumstats1.pickle")
df = df.reset_index(drop=True)
df_list.append(df)


sum_stats = pd.concat(df_list, axis=0).reset_index(drop=True)

sum_stats.to_csv("summary_stats.csv", index=False)