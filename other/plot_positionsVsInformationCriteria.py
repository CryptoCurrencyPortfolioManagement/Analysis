import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# see if the fixed portfolio allocation is due to looking change in information criteria extracted from the text

positions = pd.read_excel("./bertRNNSeperate_nn/positions_stoic-monkey-386_CV6.xlsx")
positions = positions.reset_index().melt(id_vars= ["index"])
positions["type"] = "positions"
informationcriteria = pd.read_excel("./bertRNNSeperate_nn/informationcriteria_stoic-monkey-386_CV6.xlsx")
informationcriteria = informationcriteria.reset_index().melt(id_vars= ["index"])
informationcriteria["type"] = "information criteria"

plot_df = positions.append(informationcriteria, ignore_index= True)

g = sns.FacetGrid(plot_df, row="type", hue= "variable", sharey= "row", aspect= 3)
g.map(sns.lineplot, "index", "value")
plt.suptitle("Positions vs. Information Criteria in the final Evaluation Period")
plt.tight_layout()
plt.savefig("./bertRNNSeperate_nn/positionsVsInformationCriterium.png")
plt.close()