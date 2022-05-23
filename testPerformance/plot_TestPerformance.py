import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

xls = pd.ExcelFile("./performanceTest.xlsx")
df1 = pd.read_excel(xls, 'best_seperate')
df2 = pd.read_excel(xls, 'best_wholeRun')
df3 = pd.read_excel(xls, 'Comparison')


#g = sns.FacetGrid(df1, row="measure", sharey= "row", aspect= 3)
#g.map(sns.lineplot, "index", "value")
#plt.suptitle("          Performance on Test Data")
#plt.tight_layout()
#plt.savefig("./performanceOnTestData.png")
#plt.close()

# plot graph for performance on complete time series (test + eval)
g = sns.FacetGrid(df2, row="measure", sharey= "row", aspect= 3)
g.map(sns.lineplot, "index", "value")
plt.suptitle("          Performance on All Data")
plt.tight_layout()
plt.savefig("./performanceOnAllData.png", transparent=True)
plt.close()

df2 = df2.loc[df2["index"] >=8]

# plot graph for performance on test data
g = sns.FacetGrid(df2, row="measure", sharey= "row", aspect= 3)
g.map(sns.lineplot, "index", "value")
plt.suptitle("          Performance on All Data")
plt.tight_layout()
plt.savefig("./performanceOnTestData.png")
plt.close()

# plot graph for performance on test data between different portfolio construction algos
df3 = df3.iloc[:-1, :]
df3 = df3.melt(id_vars= ["index", "measure"])
g = sns.FacetGrid(df3, row="variable", sharey= True, aspect= 3)
g.map(sns.lineplot, "index", "value")
plt.suptitle("          Test set performance")
plt.tight_layout()
plt.savefig("./performanceTestDataVergleichAlgos.png", transparent=True)
plt.close()

