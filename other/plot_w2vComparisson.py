import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("./perfComparison_w2vBased.xlsx")

df = df.iloc[:-1, 1:]

# see how different w2v models perform compared to each other
sns.boxplot(data= df.melt(), x="variable", y="value").set(title='Performance Top10 runs')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("./perfComparison_w2vBased.png")