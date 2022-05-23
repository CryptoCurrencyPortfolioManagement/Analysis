import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

# create just a plot of normalized ass3et prices
ccs = ['btc', 'eth', 'xrp', 'xem', 'etc', 'ltc', 'dash', 'xmr', 'strat', 'xlm']

price_df = pq.ParquetDataset("../data/Price.parquet", validate_schema=False, filters=[('cc', 'in', ccs)])
price_df = price_df.read(columns=["price"]).to_pandas()
price_df = price_df.pivot(index='date', columns='cc', values='price')
price_df = price_df.reset_index()
price_df["date"] = pd.to_datetime(price_df["date"]).apply(lambda x: x.date())

def normalizeCcColumns(column):
    if column.dtype == float:
        return column/column.mean()
    else:
        return column

price_df = price_df.apply(normalizeCcColumns, axis= 0)

sns.lineplot(data= price_df.melt(id_vars= "date"), x= "date", y= "value", hue="cc")
plt.title("Normalised CC prices between 01.06.2017 - 31.05.2021")
plt.xticks(rotation= 90)
plt.tight_layout()
plt.savefig("plot_CC_normalised_prices_20170601-20210531")