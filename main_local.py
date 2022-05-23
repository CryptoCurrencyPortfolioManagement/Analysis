import os
import random
import string
import ast
from tqdm import tqdm
from decimal import Decimal

import numpy as np
import pandas as pd
import scipy.stats as stats
import pyarrow.parquet as pq

import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_runs_for_project, get_runs_for_sweep, get_history_of_run

import statsmodels.formula.api as smf

from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector as selector

from gdriveHandler import searchFiles

from empyrical import cum_returns

def stats_sharpe(row):
    return {"med": row["sharpeRatio_med"],
            "mean": row["sharpeRatio"],
            "q1": row["sharpeRatio_q1"],
            "q3": row["sharpeRatio_q3"],
            "whislo": row["sharpeRatio_min"],
            "whishi": row["sharpeRatio_max"]}

def removeExtremeOutliersByGroup(group, column, max_z = 3):
    Q1 = group[metric].quantile(0.25)
    Q3 = group[metric].quantile(0.75)
    IQR = Q3 - Q1
    group["outlier"] = ((group[metric] < (Q1 - 3 * IQR)) | (group[metric] > (Q3 + 3 * IQR)))
    return group

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

#projects_base_names = [
#    "dictRnnApproaches_final",
#    "nnApproaches_final",
#    "w2vsumRnnApproaches_final",
#    "w2vRnnApproaches_final",
#    "bertRNNSeperateApproaches_final",
#    "bertRnnApproaches_final",
#    "transformerStackApproaches_final",
#    "transformerRnnApproaches_final"
#]

#projects_sweep_names = {
#    "dictRnnApproaches_final": "gbafpdxe",
#    "nnApproaches_final": "hhidikec",
#    "w2vsumRnnApproaches_final": "hxr1ulfj",
#    "w2vRnnApproaches_final": "zz149zoz",
#    "bertRNNSeperateApproaches_final": "9azvhzrd",
#    "bertRnnApproaches_final": "fo9fkcvy",
#    "transformerStackApproaches_final": "07a210dl",
#    "transformerRnnApproaches_final": "svzj9jmo"
#}

results = []

metrics = [
    "mdd",
    "cumReturn",
    "calmarRatio",
    "sortinoRatio",
    "sharpeRatio"
]

tops = [50, 30, 10, 5, 1]

ccs = ['btc', 'eth', 'xrp', 'xem', 'etc', 'ltc', 'dash', 'xmr', 'strat', 'xlm']


# creates a plot showing the positions hold by the algorithms as well as the cumulative returns of the strategy
def position_plots(labels, position_history, returns_history, name, total_return, total_sharpeRatio):
    def _plot_position(ax, positions):
        for i, label in enumerate(labels):
            ax.plot(positions[:, i], label=label)
        ax.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        ax.title.set_text('Positions')

    def _plot_cumreturns(ax, returns):
        ax.plot(returns, label="Cumulative Returns")
        ax.title.set_text("Cumulative Returns")
        ax.set_ylim([-1.05, 1.05])

    fig, (ax1, ax2) = plt.subplots(2, 1)
    _plot_position(ax1, position_history)
    _plot_cumreturns(ax2, returns_history)

    fig.suptitle(
        "{} \n".format(name) +
        "Total Reward: %.6f" % total_return + ' ~ ' +
        "Total Sharpe: %.6f" % total_sharpeRatio
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

sources = [("price regression", "data/localApproaches/Overview_autoregpriceonlyApproaches.xlsx"),
           ("dict and price regression", "data/localApproaches/Overview_autoregApproaches.xlsx")]

for project_base_name, source in sources:
    print(source)
    #df = get_runs_for_project(project_base_name)
    df = pd.read_excel(source)

    if project_base_name == "price regression":
        df["name"] = df["shared_model"] + "_" + df["window_size"].apply(lambda x: str(x))
    elif project_base_name == "dict and price regression":
        df["name"] = df["feature"] + "_" + df["shared_model"] + "_" + df["source"] + "_" + df["use_price"].apply(lambda x: {1: "True", 0: "False"}[x]) + "_" + df["window_size"].apply(lambda x: str(x))

    df = df.set_index("name")

    for metric in metrics:
        df[metric] = df[metric].astype(float)
        df = df.loc[(df[metric] < 1000000) & (df[metric] > -1000000)]
    df = df.replace([np.inf, -np.inf], np.nan)

    #create necessary directories
    if not os.path.exists("./output/{}".format(project_base_name.replace("Approaches_final", ""))):
        os.mkdir("./output/{}".format(project_base_name.replace("Approaches_final", "")))

    #analysis
    independent_variables = [column for column in df.columns if not any([metric in column for metric in metrics]) and column[0] != "_"]
    independent_variables.sort()

    # influence of hyper parameters
    for metric in metrics:
        sns.histplot(data=df, x=metric, common_norm=False).set(title='{}: distribution of {}'.format(project_base_name.replace("Approaches_final", ""), metric))
        plt.xlim(-2, 3)
        plt.savefig("./output/{}/plot_{}.png".format(project_base_name.replace("Approaches_final", ""), metric))
        plt.close()
        for independent_variable in independent_variables:
            if df[independent_variable].nunique() < 10:
                sns.kdeplot(data=df, x=metric, hue=independent_variable, common_norm=False).set(title='{}: distribution of {} by {}'.format(project_base_name.replace("Approaches_final", ""), metric, independent_variable))
            else:
                bins = np.histogram_bin_edges(df[independent_variable], bins='auto')
                metric_avgs = []
                for i in range(len(bins)-1):
                    if i < len(bins):
                        metric_avgs.append(df.loc[(df[independent_variable] > bins[i]) & (df[independent_variable] < bins[i + 1]), metric].mean())
                    else:
                        metric_avgs.append(df.loc[df[independent_variable] > bins[i], metric].mean())
                bins = bins[:-1] + 0.5* (bins[1:]-bins[:-1])
                sns.lineplot(data= pd.DataFrame(columns=[metric, independent_variable], data= np.transpose(np.array([metric_avgs, bins]))), x= independent_variable, y= metric).set(title='{}: average {} vs. {}'.format(project_base_name.replace("Approaches_final", ""), metric, independent_variable))
            plt.savefig("./output/{}/plot_{}_{}.png".format(project_base_name.replace("Approaches_final", ""), metric, independent_variable))
            plt.close()

    # overview top 50, 30, 10, 5
    df = df.sort_values(by= "sharpeRatio", ascending= False)

    df_for_boxplots = pd.DataFrame()
    for top in tops:
        temp = df.iloc[:top].copy()
        temp["selected top (sorted by sharpe ratio)"] = top
        df_for_boxplots = df_for_boxplots.append(temp)
    for metric in metrics:
        # remove extreme outliers
        metric_std = df_for_boxplots[metric].std()
        metric_mean = df_for_boxplots[metric].mean()
        sns.boxplot(data= df_for_boxplots.loc[(df_for_boxplots[metric] < metric_mean + 3 * metric_std) & (df_for_boxplots[metric] > metric_mean - 3 * metric_std)], x="selected top (sorted by sharpe ratio)", y=metric).set(title='Performance {}'.format(project_base_name.replace("Approaches_final", "")))
        plt.savefig("./output/{}/plot_performance_{}.png".format(project_base_name.replace("Approaches_final", ""), metric))
        plt.close()
    del df_for_boxplots

    # see impact of measures and significance
    categorical_variables_selector = selector(dtype_include=object)
    non_categorical_variables = list(set(independent_variables) - set(categorical_variables_selector(df[independent_variables])))
    df_analysis = df.copy()
    df_analysis[non_categorical_variables] = StandardScaler().fit_transform(df_analysis[non_categorical_variables])

    for metric in metrics:
        mod = smf.ols(formula='{} ~ {} + {}'.format(metric, " + ".join(independent_variables), " + ".join(["np.power({}, 2)".format(x) for x in non_categorical_variables if x not in ["use_attention", "use_price"]])), data=df_analysis)
        #mod = smf.ols(formula='{} ~ {}'.format(metric, " + ".join(independent_variables)), data=df_analysis)

        res = mod.fit()
        with open("./output/{}/table_significance_{}.tex".format(project_base_name.replace("Approaches_final", ""), metric), "w") as f:
            f.write(res.summary().as_latex())
        res_df = pd.DataFrame(res.summary().tables[1].data[1:], columns=res.summary().tables[1].data[0])
        res_df = res_df.set_index("").astype(float)
        res_df["sgnf_stars"] = ""
        for sgnf_level, descriptor in zip([0.1, 0.05, 0.01, 0.001], [".", "*", "**", "***"]):
            res_df.loc[res_df["P>|t|"] < sgnf_level, "sgnf_stars"] = descriptor
        res_df.to_csv("./output/{}/table_significance_{}.csv".format(project_base_name.replace("Approaches_final", ""), metric))
        #plt.rc('figure', figsize=(10, 10))
        #plt.text(0.01, 0.05, str(res.summary()), {'fontsize': 10}, fontproperties='monospace')
        #plt.axis('off')
        #plt.tight_layout()
        #plt.savefig("./output/{}/table_significance_{}.png".format(project_base_name, metric))
        #plt.close()
    del df_analysis

    # analyse the detailed data of the top5
    top_to_use = 10
    top_history_data= pd.DataFrame()
    top10_performance_data = []
    for i in range(top_to_use):
        sample = df.iloc[i]
        #idStr_list = [float(x) if abs(Decimal(str(x)).as_tuple().exponent) > 1 else int(x) for x in sample[independent_variables].tolist()]
        idStr_list = []
        for x in sample[independent_variables].tolist():
            if isinstance(x, int) or isinstance(x, float):
                if abs(Decimal(str(x)).as_tuple().exponent) > 1:
                    idStr_list.append(float(x))
                else:
                    idStr_list.append(int(x))
            else:
                idStr_list.append(x)

        rand_name = randomword(15)
        for j in range(0, 7):
            raw_data = pd.read_excel("data/localApproaches/approaches/{}_CV{}.xlsx".format(sample.name, j))
            if len(raw_data) > 0:
                temp = raw_data.rename(columns= {"total_return": "cumReturn"}).iloc[-1].loc[metrics]
                if len(sample.name) > 15:
                    temp["run"] = rand_name
                else:
                    temp["run"] = sample.name
                top10_performance_data.append(temp)

        raw_data = pd.read_excel("data/localApproaches/approaches/{}_CV6.xlsx".format(sample.name))
        if len(raw_data) > 0:
            position_array = raw_data["positions"].tolist()
            position_array = [x.replace("\n", "").replace("[", "").replace("]", "").split() for x in position_array]
            position_array = np.array(position_array).astype(float)
            position_plots(ccs, position_array, raw_data["total_return"], "{} - top{}".format(project_base_name.replace("Approaches_final", ""), i+1), sample["cumReturn"], sample["sharpeRatio"])
            plt.savefig("./output/{}/plot_positions_CV6_top{}.png".format(project_base_name.replace("Approaches_final", ""), i+1))
            plt.close()
        else:
            print("no data found ({}/{} - top{})".format(project_base_name, sample.name, i))

    for metric in metrics:
        sns.boxplot(data= pd.DataFrame(top10_performance_data), x="run", y=metric).set(title='Performance Top10 runs {}'.format(project_base_name.replace("Approaches_final", "")))
        plt.ylim(-3, 5)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('./output/{}/plot_{}_top10_runs.png'.format(project_base_name, metric))
        plt.close()

    #compare movement to text volume
    warm_up_window = 180
    rolling_windows_size = 360
    rolling_windows_shift = 90
    train_start_date = pd.to_datetime("2017-06-01").date()
    train_end_date = pd.to_datetime("2020-05-31").date()

    date_range = [x.date() for x in pd.date_range(train_start_date, train_end_date)]
    windows = []
    window_start_idx = warm_up_window
    window_end_idx = warm_up_window + rolling_windows_size
    while window_end_idx <= len(date_range):
        windows.append((date_range[window_start_idx], date_range[window_end_idx]))
        window_start_idx += rolling_windows_shift
        window_end_idx = window_start_idx + rolling_windows_size

    if "source" in df.columns:
        text_df = pq.ParquetDataset("data/Content.parquet", validate_schema=False, filters=[('cc', 'in', ccs), ('source', '=', df.iloc[0]["source"])])
        text_df = text_df.read(columns=["content_raw"]).to_pandas()
        text_df["date"] = pd.to_datetime(text_df["date"]).apply(lambda x: x.date())

        content_per_test = []
        content_per_train = []
        for window in windows:
            content_per_test.append(sum((text_df["date"] > window[0]) & (text_df["date"] < window[1])))
            content_per_train.append(sum(text_df["date"] < window[0]))
        runs_in_history_data = top_history_data.loc[top_history_data["source"] == df.iloc[0]["source"]]["run"].nunique()
        plot_df = top_history_data.loc[top_history_data["source"] == df.iloc[0]["source"], ["index", "run", "sharpeRatio_CV"]]
        plot_df = plot_df.rename(columns = {'sharpeRatio_CV':'value'})
        plot_df["variable"] = "sharpeRatio per run"
        avg_top_history_data = top_history_data.loc[top_history_data["source"] == df.iloc[0]["source"]].loc[:, ["index", "sharpeRatio_CV"]].groupby("index").mean()["sharpeRatio_CV"].tolist()
        plot_df = plot_df.append(pd.DataFrame(data=map(list, zip(*[list(range(len(avg_top_history_data))),
                                                                   ["all"] * len(avg_top_history_data),
                                                                   avg_top_history_data,
                                                                   ["avg sharpeRatio"] * len(avg_top_history_data)])),
                                              columns=["index", "run", "value", "variable"]), ignore_index=True)
        plot_df = plot_df.append(pd.DataFrame(data= map(list, zip(*[list(range(len(content_per_test))),
                                                                    ["all"]*len(content_per_test) ,
                                                                    content_per_test,
                                                                    ["texts in test set"]*len(content_per_test)])),
                                              columns=["index", "run", "value", "variable"]), ignore_index= True)
        plot_df = plot_df.append(pd.DataFrame(data=map(list, zip(*[list(range(len(content_per_train))),
                                                                   ["all"] * len(content_per_train),
                                                                   content_per_train,
                                                                   ["texts in train set"] * len(content_per_train)])),
                                              columns=["index", "run", "value", "variable"]), ignore_index=True)
        g = sns.FacetGrid(plot_df, row="variable", hue= "run", sharey= "row")
        g.map(sns.lineplot, "index", "value")
        g.fig.suptitle('{}:\n performance vs available data\n ({} runs)\n'.format(project_base_name.replace("Approaches_final", ""), runs_in_history_data))
        plt.tight_layout()
        plt.savefig("./output/{}/plot_performanceVsAvailableData_avg{}.png".format(project_base_name.replace("Approaches_final", ""), top_to_use))
        plt.close()

        temp = plot_df.loc[plot_df["run"] == "all", ["index", "variable", "value"]].pivot(index=["index"], columns= "variable")
        temp.columns = [x[1] for x in temp.columns]
        temp = temp.drop("avg sharpeRatio", axis= 1)
        plot_df = plot_df.loc[plot_df["run"] != "all"].drop(["variable"], axis= 1).join(temp, on= "index", how= "left")
        plot_df = plot_df.rename({"value": "sharpeRatio"}, axis= 1)

        plot_df.columns = [x.replace(" ", "_") for x in plot_df.columns]
        plot_df[["texts_in_test_set", "texts_in_train_set"]] = StandardScaler().fit_transform(plot_df[["texts_in_test_set", "texts_in_train_set"]])
        mod = smf.ols(formula='sharpeRatio ~ texts_in_test_set + texts_in_train_set', data= plot_df)
        res = mod.fit()
        res_df = pd.DataFrame(res.summary().tables[1].data[1:], columns=res.summary().tables[1].data[0])
        res_df = res_df.set_index("").astype(float)
        res_df["sgnf_stars"] = ""
        for sgnf_level, descriptor in zip([0.1, 0.05, 0.01, 0.001], [".", "*", "**", "***"]):
            res_df.loc[res_df["P>|t|"] < sgnf_level, "sgnf_stars"] = descriptor
        res_df.to_csv("./output/{}_table_significance_performanceVsAvailableData_avg{}.csv".format(project_base_name.replace("Approaches_final", ""), top_to_use))

        # plot against market trend
        price_df = pq.ParquetDataset("./data/Price.parquet", validate_schema=False, filters=[('cc', 'in', ccs)])
        price_df = price_df.read(columns=["price"]).to_pandas()
        price_df = price_df.pivot(index='date', columns='cc', values='price')
        price_df = price_df.reset_index()
        price_df["date"] = pd.to_datetime(price_df["date"]).apply(lambda x: x.date())
        price_in_window = []
        for window in windows:
            temp = price_df.loc[(price_df["date"] > window[0]) & (price_df["date"] < window[1])].drop("date", axis= 1).pct_change(1).iloc[1:]
            temp = temp.apply(lambda x: cum_returns(x), axis= 0).iloc[-1]
            price_in_window.append(temp.mean())
        plot_df = top_history_data.loc[top_history_data["source"] == df.iloc[0]["source"], ["index", "run", "sharpeRatio_CV"]]
        plot_df = plot_df.rename(columns={'sharpeRatio_CV': 'value'})
        plot_df["variable"] = "sharpeRatio per run"
        avg_top_history_data = top_history_data.loc[top_history_data["source"] == df.iloc[0]["source"]].loc[:,
                               ["index", "sharpeRatio_CV"]].groupby("index").mean()["sharpeRatio_CV"].tolist()
        plot_df = plot_df.append(pd.DataFrame(data=map(list, zip(*[list(range(len(avg_top_history_data))),
                                                                   ["all"] * len(avg_top_history_data),
                                                                   avg_top_history_data,
                                                                   ["avg sharpeRatio"] * len(avg_top_history_data)])),
                                              columns=["index", "run", "value", "variable"]), ignore_index=True)
        plot_df = plot_df.append(pd.DataFrame(data=map(list, zip(*[list(range(len(price_in_window))),
                                                                   ["all"] * len(price_in_window),
                                                                   price_in_window,
                                                                   ["market trend"] * len(price_in_window)])),
                                              columns=["index", "run", "value", "variable"]), ignore_index=True)
        g = sns.FacetGrid(plot_df, row="variable", hue="run", sharey="row")
        g.map(sns.lineplot, "index", "value")
        g.fig.suptitle('{}:\n performance vs market trend\n ({} runs)\n'.format(project_base_name.replace("Approaches_final", ""), runs_in_history_data))
        plt.tight_layout()
        plt.savefig("./output/{}/plot_performanceVsMarketTrend_avg{}.png".format(project_base_name.replace("Approaches_final", ""), top_to_use))
        plt.close()

        # add training data to controll for available traning data
        plot_df = plot_df.append(pd.DataFrame(data=map(list, zip(*[list(range(len(content_per_train))),
                                                                   ["all"] * len(content_per_train),
                                                                   content_per_train,
                                                                   ["texts in train set"] * len(content_per_train)])),
                                              columns=["index", "run", "value", "variable"]), ignore_index=True)

        temp = plot_df.loc[plot_df["run"] == "all", ["index", "variable", "value"]].pivot(index=["index"], columns="variable")
        temp.columns = [x[1] for x in temp.columns]
        temp = temp.drop("avg sharpeRatio", axis=1)
        plot_df = plot_df.loc[plot_df["run"] != "all"].drop(["variable"], axis=1).join(temp, on="index", how="left")
        plot_df = plot_df.rename({"value": "sharpeRatio"}, axis=1)

        plot_df.columns = [x.replace(" ", "_") for x in plot_df.columns]
        plot_df[["market_trend"]] = StandardScaler().fit_transform(
            plot_df[["market_trend"]])
        mod = smf.ols(
            formula='sharpeRatio ~ market_trend + texts_in_train_set',
            data=plot_df)
        res = mod.fit()
        res_df = pd.DataFrame(res.summary().tables[1].data[1:], columns=res.summary().tables[1].data[0])
        res_df = res_df.set_index("").astype(float)
        res_df["sgnf_stars"] = ""
        for sgnf_level, descriptor in zip([0.1, 0.05, 0.01, 0.001], [".", "*", "**", "***"]):
            res_df.loc[res_df["P>|t|"] < sgnf_level, "sgnf_stars"] = descriptor
        res_df.to_csv("./output/{}_table_significance_performanceVsMarketTrend_avg_{}.csv".format(project_base_name.replace("Approaches_final", ""), top_to_use))

        results.append({"maxSR": df["sharpeRatio"].max(), "name": project_base_name, "data": df})