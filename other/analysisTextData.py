import numpy as np
import pandas as pd
import scipy.stats as stats
import pyarrow.parquet as pq

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS


# retrieve data
ccs = ['btc', 'eth', 'xrp', 'xem', 'etc', 'ltc', 'dash', 'xmr', 'strat', 'xlm']
stopwords = set(STOPWORDS.union(set(ccs)))
eval_start_date = pd.to_datetime("2017-11-28").date()
eval_end_date = pd.to_datetime("2020-05-16").date()
test_start_date = pd.to_datetime("2019-08-20").date()
test_end_date = pd.to_datetime("2021-05-11").date()

text_df = pq.ParquetDataset("../data/Content.parquet", validate_schema=False, filters=[('cc', 'in', ccs), ('source', '=', "CoinTelegraph")])
text_df = text_df.read(columns=["content_processed"]).to_pandas()
text_df["date"] = pd.to_datetime(text_df["date"]).apply(lambda x: x.date())
text_df = text_df.sort_values(by= "date", ascending= False)

def groupTogether(group):
    temp = pd.Series()
    temp["ccs"] = group["cc"].unique().to_list()
    return temp
text_df = text_df.groupby(["content_processed", "date"]).apply(groupTogether).reset_index()

text_df["dataset"] = np.nan
text_df.loc[(text_df["date"] >= eval_start_date) & (text_df["date"] <= eval_end_date) ,"dataset"] = "Evaluation set"
text_df.loc[(text_df["date"] >= test_start_date) & (text_df["date"]  <= test_end_date),"dataset"] = "Test set"
text_df = text_df.dropna(subset=["dataset"])

text_df["text length"] = text_df["content_processed"].apply(lambda x: len(x))
text_df["num words"] = text_df["content_processed"].apply(lambda x: len(x.split(" ")))
text_df["num ccs"] = text_df["ccs"].apply(lambda x: len(x))

# plot different text characteristics evaluation period vs. testing period
for variable in ["text length", "num words", "num ccs"]:
    sns.violinplot(data= text_df, x= "dataset", y=variable, hue_order= ["Test set", "Evaluation set"])
    plt.title("{}: Evaluation vs Test set".format(variable))
    plt.tight_layout()
    plt.savefig("./EvalVsTestSet/EvalvsTestSet_{}.png".format(variable))
    plt.close()

# plot grouped violin plots of text length
sns.violinplot(data=text_df.groupby(["date", "dataset"])["content_processed"].count().reset_index(), x="dataset", y="content_processed", hue_order= ["Test set", "Evaluation set"])
plt.title("{}: Evaluation vs Test set".format("Texts per day"))
plt.tight_layout()
plt.savefig("./EvalVsTestSet/EvalvsTestSet_{}.png".format("Texts per day"))
plt.close()

# create wordcouds for words in eval and test set
for dataset in text_df["dataset"].unique():
    wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(" ".join(text_df["content_processed"].to_list()))

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("WordCloud {}".format(dataset))
    plt.tight_layout()
    plt.savefig("./EvalVsTestSet/wordCloud_{}.png".format(dataset))
    plt.close()