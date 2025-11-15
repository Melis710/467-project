import pandas as pd
df = pd.DataFrame({'missing_steps': [(), (1,), (2,), (1, 2)], 'is_correct': [0.8, 0.7, 0.6, 0.5]})
all_subsets_missing = [(), (1,), (2,), (1, 2)]
mask=df['missing_steps'].isin(all_subsets_missing)
v_S=zip(all_subsets_missing,df[mask])
for item in v_S:
    print(item)


v_S = {subset: df.loc[df["missing_steps"] == subset, "is_correct"].values[0] for subset in all_subsets_missing}
print(v_S)



# DENEMEEEEEEEEEEE