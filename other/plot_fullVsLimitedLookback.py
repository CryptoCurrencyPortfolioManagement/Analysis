import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("./fullVsLimitedLookback.xlsx")
df = df.rename(columns= {"Unnamed: 0": "index"})

# compare a algorithm with complete lookback and with lookback of limited length
g = sns.FacetGrid(df, aspect= 3)
#sns.lineplot(data= df.melt(id_vars="index"), x="index", y="value", hue="variable")
g.map(sns.lineplot, data= df.melt(id_vars="index"), x="index", y="value", hue="variable")
plt.suptitle("Performance full vs limited (450 days) lookback")
plt.tight_layout()
plt.savefig("./fullVsLimitedLookback.png")