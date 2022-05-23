import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("./modelComplexityVsPerformance.xlsx")

df = df.sort_values("Number of parameters")

# plot complexity of model vs. its performance
sns.scatterplot(data=df, x="Number of parameters", y= "sharpeRatio", hue="model")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("./modelComplexityVsPerformance.png")
plt.close()

df = df.iloc[:-3, :]
sns.scatterplot(data=df, x="Number of parameters", y= "sharpeRatio", hue="model")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("./modelComplexityVsPerformance_zoomedIn.png")
plt.close()