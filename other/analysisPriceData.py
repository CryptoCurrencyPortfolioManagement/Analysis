import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.weightstats import ztest as ztest
import pyarrow.parquet as pq

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS


ccs = ['btc', 'eth', 'xrp', 'xem', 'etc', 'ltc', 'dash', 'xmr', 'strat', 'xlm']

# generate the time windows
warm_up_window = 180
rolling_windows_size = 360
rolling_windows_shift = 90
train_start_date = pd.to_datetime("2017-06-01").date()
train_end_date = pd.to_datetime("2021-05-31").date()

date_range = [x.date() for x in pd.date_range(train_start_date, train_end_date)]
windows = []
window_start_idx = warm_up_window
window_end_idx = warm_up_window + rolling_windows_size
while window_end_idx <= len(date_range):
    windows.append((date_range[window_start_idx], date_range[window_end_idx]))
    window_start_idx += rolling_windows_shift
    window_end_idx = window_start_idx + rolling_windows_size

# retrieve price data
price_df = pq.ParquetDataset("../data/Price.parquet", validate_schema=False, filters=[('cc', 'in', ccs)])
price_df = price_df.read(columns=["price"]).to_pandas()
price_df = price_df.pivot(index='date', columns='cc', values='price')
price_df = price_df.reset_index()
price_df["date"] = pd.to_datetime(price_df["date"]).apply(lambda x: x.date())
price_df = price_df.set_index("date")
price_df[:] = price_df.values / price_df.shift(periods=1).values - 1
price_df = price_df.reset_index()

# subdivide price data based on windows
price_in_window = []
price_df["window"] = np.nan
for i, window in enumerate(windows):
    price_df.loc[(price_df["date"] > window[0]) & (price_df["date"] < window[1]), "window"] = i+1
    temp = price_df.loc[(price_df["date"] > window[0]) & (price_df["date"] < window[1])].drop("date", 1).iloc[1:]
    price_in_window.append(temp)
long_price_df = price_df.melt(id_vars= ["window", "date"]).dropna()

# create boxplots
sns.boxplot(data= long_price_df, y= "value", x= "window")
plt.title("Distribution of returns between evaluation and test sets")
plt.tight_layout()
plt.savefig("./analysisPriceData.png")
plt.close()

pricedf_mean = long_price_df.groupby("window").mean()
pricedf_std = long_price_df.groupby("window").std()
pricedf_q1 = long_price_df.groupby("window").quantile(0.25)
pricedf_q3 = long_price_df.groupby("window").quantile(0.75)

pricedf_stats = pd.DataFrame({"mean": pricedf_mean["value"].to_list(), "std": pricedf_std["value"].to_list(), "Q1": pricedf_q1["value"].to_list(), "Q3": pricedf_q3["value"].to_list()})
pricedf_stats.to_excel("./analysisPriceData.xlsx")
pricedf_stats.to_csv("./analysisPriceData.csv")

# see if price data significantly differs between windows
pricedf_ztests = []
for i in range(len(windows)):
    i = i+1
    pricedf_ztests_i = []
    for j in range(len(windows)):
        j = j+1
        zStat, pValue = ztest(long_price_df.loc[long_price_df["window"] == i]["value"], long_price_df.loc[long_price_df["window"] == j]["value"])
        pricedf_ztests_i.append("z-test({}): z-value: {:.3f}, p-value:{:.3f}".format(long_price_df.loc[long_price_df["window"] == i].shape[0] + long_price_df.loc[long_price_df["window"] == i].shape[0] -2, zStat, pValue))
    pricedf_ztests.append(pricedf_ztests_i)

pricedf_ztests = pd.DataFrame(pricedf_ztests)
pricedf_ztests = pricedf_ztests.transpose()
pricedf_ztests.to_excel("./analysisPriceData_ztests.xlsx")
pricedf_ztests.to_csv("./analysisPriceData_ztests.csv")

print(price_df)
