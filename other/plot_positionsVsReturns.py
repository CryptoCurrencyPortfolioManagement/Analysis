import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# analyse how different methods of creating position react to changes in asset returns
currency = "btc"

df = pd.read_excel("./ComparisonPositionsVsReturns_CV6_{}.xlsx".format(currency))
df = df.dropna(axis= "index")
sample_len = len(df)

def get_perfInd(row):
    for index in row.index[1:]:
        if row[index] / row[0] > 0:
            row[index] = "green"
        else:
            row[index] = "red"
    return row

df_perfInd = df.copy().apply(get_perfInd, axis= 1)
df_perfInd[currency] = "blue"
df_perfInd_bg = df_perfInd.copy()
df_perfInd_bg[:] = "grey"
del df_perfInd_bg[currency]
df_perfInd = df_perfInd_bg.append(df_perfInd)

df = df.melt()
df_perfInd = df_perfInd.melt()

def get_id(row):
    variable = row["variable"] + "_" + str(row["index"] % sample_len)
    return variable

df["merge_variable"] = df.copy().reset_index().apply(get_id, axis= 1)
df_perfInd["merge_variable"] = df_perfInd.copy().reset_index().apply(get_id, axis= 1)
df_perfInd = df_perfInd.dropna(subset= ["value"])

plot_df = pd.merge(df, df_perfInd, how= "left", on= "merge_variable", suffixes= ("_positions", "_perfIndicator"))
plot_df.index = plot_df.index % sample_len

def myplot(arg1, **kwargs):
    if kwargs["label"] == "blue":
        return sns.lineplot(x=arg1.index, y=arg1)
    elif kwargs["label"] == "grey":
        return sns.lineplot(x= arg1.index, y= arg1, color= kwargs["label"], linewidth = 0.7)
    else:
        return sns.scatterplot(x=arg1.index, y=arg1, color=kwargs["label"])
        #df = pd.DataFrame((arg1, arg2)).transpose().reset_index(drop= True)
        #df = df.rename(columns={"value_perfIndicator": "value_color"})
        #fig, ax = plt.subplots()
        #for idx, row in df.iloc[:-1, :].iterrows():
        #    # print(idx, row.value, row.colors)
        #    ax = sns.lineplot(x= [idx, idx + 1], y= [row.value_positions, df.loc[idx + 1, 'value_positions']], color=row.value_color, ax= ax)
        #ax.set_ylim(0, 1)
        #return ax

g = sns.FacetGrid(plot_df, row="variable_positions", sharey="row", hue="value_perfIndicator", height=1.7, aspect=4)
g.map(myplot, "value_positions")
plt.savefig("./ComparisonPositionsVsReturns_CV6_{}.png".format(currency))