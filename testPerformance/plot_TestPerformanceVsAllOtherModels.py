import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Eval set performance
df = pd.read_csv("./rb-bertRNNSeperate_allruns.csv")
df = df.loc[~df["sharpeRatio"].isnull(), ["sharpeRatio", "mdd", "cumReturn"]]
df["type"] = "evaluation set performances"

# Test set performance
testPerf = {"sharpeRatio": [-0.64495], "mdd": [-0.53955], "cumReturn": [-0.4129425], "type": ["test set performance"]}
df = df.append(pd.DataFrame(testPerf))

# plots display performance of test set vs performance on the different evaluation sets
sns.boxplot(data= df, y="sharpeRatio", x="type")
plt.suptitle("Performance Evaluation set vs Test set")
plt.tight_layout()
plt.savefig("./performanceTestVsEvaluationSet.png")
plt.close()

sns.swarmplot(data= df, y="sharpeRatio", x="type")
plt.axhline(testPerf["sharpeRatio"][0], color= "#E1812C")
plt.suptitle("Performance Evaluation set vs Test set")
plt.tight_layout()
plt.savefig("./performanceTestVsEvaluationSetSwarm.png")
plt.close()