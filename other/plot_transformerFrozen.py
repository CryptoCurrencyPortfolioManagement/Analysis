import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

xls = pd.ExcelFile("./perfComparison_transformerFrozen.xlsx")
df1 = pd.read_excel(xls, 'Sheet1')
df2 = pd.read_excel(xls, 'Sheet2')

df1 = df1.iloc[:, 1:]
df2 = df2.iloc[:, 1:]

# compare best runs with frozen and unfrozen transformers via plots
sns.boxplot(data= df1.melt(), x="variable", y="value").set(title='Performance Top10 runs unfrozen')
plt.xticks(rotation=90)
plt.ylim(0.5, 2)
plt.tight_layout()
plt.savefig("./perfComparison_TransformerUnfrozen.png")
plt.close()

sns.boxplot(data= df2.melt(), x="variable", y="value").set(title='Performance Top10 runs frozen')
plt.xticks(rotation=90)
plt.ylim(0.5, 2)
plt.tight_layout()
plt.savefig("./perfComparison_TransformerFrozen.png")
plt.close()