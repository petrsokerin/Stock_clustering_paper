from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import (davies_bouldin_score,
                            silhouette_score,
                            calinski_harabasz_score,
                            homogeneity_score)

from utils.portfolio import MarkowitzPortfolio
from utils.portfolio_metrics import (find_max_recovery, find_max_drawdown)

import warnings
warnings.filterwarnings("ignore")


def make_generator(parameters):
    """generator for Grid Search for clustering model"""
    if not parameters:
        yield dict()
    else:
        key_to_iterate = list(parameters.keys())[0]
        next_round_parameters = {p: parameters[p]
                                 for p in parameters if p != key_to_iterate}
        for val in parameters[key_to_iterate]:
            for pars in make_generator(next_round_parameters):
                temp_res = pars
                temp_res[key_to_iterate] = val
                yield temp_res


class ClusteringGridSearch:
    """Class provide grid search procedure for clustering task"""

    def __init__(self, estimator, param_grid, scoring):

        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring

        self.best_params_ = dict()
        self.best_estimator_ = None
        self.best_score_ = - 1e+6

    def fit(self, X):
        all_params = self.estimator.get_params()

        for params in make_generator(self.param_grid):

            all_params.update(params)
            self.estimator = self.estimator.set_params(**all_params)
            score = self.scoring(self.estimator, X)

            if score > self.best_score_:
                self.best_score_ = score
                self.best_estimator_ = self.estimator.fit(X)
                self.best_params_ = params


def get_clusters(data,
                 tickers_list,
                 clust_model,
                 make_grid=False,
                 grid_params=None,
                 grid_metric=None):
    """providing clustering procedure, return labels for every stock"""

    if make_grid:
        grid_model = ClusteringGridSearch(estimator=clust_model, param_grid=grid_params,
                                          scoring=grid_metric)
        grid_model.fit(data)
        clust_model = grid_model.best_estimator_
    else:
        clust_model.fit(data)
    df_clusters = pd.DataFrame([tickers_list, clust_model.labels_], index=['ticker', 'cluster']).T
    return df_clusters


def select_assets(df_clusters, df_pct, selection_method, n_save=2, **kargs):
    """asset selection procedure according with selection method for forming financial portfolio
    return list of selected tickers"""

    selected_tickers = []
    for cluster in np.unique(df_clusters['cluster'].values):
        df_clusters_loc = df_clusters[df_clusters['cluster'] == cluster]
        list_tickers = df_clusters_loc['ticker'].values.tolist()
        selected_tickers_loc = selection_method(list_tickers, n_save=n_save, df_pct=df_pct, **kargs)
        selected_tickers.extend(selected_tickers_loc)

    return selected_tickers


def get_train_test_data(df_pct,
                        test_start_per,
                        window_train,
                        window_test):
    """sub function for grid serach testing, providing train-test split procedure for every period"""

    # slicing data train
    train_finish_per = test_start_per
    train_start_per = train_finish_per - window_train

    train_year_start_per = train_start_per // 12
    train_month_start_per = train_start_per % 12 + 1

    train_year_finish_per = train_finish_per // 12
    train_month_finish_per = train_finish_per % 12 + 1

    mask_train = (df_pct.index > datetime(train_year_start_per, train_month_start_per, 1))
    mask_train = mask_train & (df_pct.index < datetime(train_year_finish_per, train_month_finish_per, 1))
    df_train = df_pct[mask_train]

    # slicing data test

    test_finish_per = train_finish_per + window_test

    test_year_start_per = train_year_finish_per
    test_month_start_per = train_month_finish_per

    test_year_finish_per = test_finish_per // 12
    test_month_finish_per = test_finish_per % 12 + 1

    mask_test = (df_pct.index > datetime(test_year_start_per, test_month_start_per, 1))
    mask_test = mask_test & (df_pct.index < datetime(test_year_finish_per, test_month_finish_per, 1))
    df_test = df_pct[mask_test]

    return df_train, df_test


def backtesting_one_model(df_pct,  # df with pct_changes: columns - tick, index - date
                          port_model=MarkowitzPortfolio,  # portfolio estimation function
                          window_train=24,  # size of train window in months
                          window_test=1,  # size of train window in months
                          test_start_year=2020,  # start data year
                          test_start_month=1,  # start data month
                          test_finish_year=2022,  # end data year
                          test_finish_month=11,  # end data month
                          **kargs):
    """providing backtesting for financial portfolios,
    return portfio pct_change data and weights of portfolios for every period"""

    weights_all = []
    return_portfolio = pd.DataFrame([])

    test_start_month = test_start_year * 12 + test_start_month - 1  # indexing from 0
    test_finish_month = test_finish_year * 12 + test_finish_month - 1  # indexing from 0
    train_finish_month = test_finish_month - window_train - window_test + 1

    for test_start_per in range(test_start_month, test_finish_month, window_test):
        df_train, df_test = get_train_test_data(df_pct, test_start_per, window_train, window_test)

        mu = (((df_train + 1).prod()) ** (
                    1 / len(df_train)) - 1).values * 252  # средняя доходность за год (252 раб дня)
        Sigma = df_train.cov().values * 252  # ковариационная матрица за год (252 раб дня)

        port_ = port_model(mu, Sigma, kargs=kargs)
        weights, _ = port_.fit()

        weights_all.append(weights)

        return_portfolio_loc = pd.DataFrame(df_test.values @ weights, index=df_test.index)
        return_portfolio = pd.concat([return_portfolio, return_portfolio_loc])

    return weights_all, return_portfolio


def clustering_estimation(X, labels, true_sectors):
    """calculating metrics of clustering"""

    df_labels_sectors = pd.merge(labels, true_sectors, on='ticker', how='left')

    result_dict = dict()

    result_dict['DB'] = davies_bouldin_score(X, df_labels_sectors['cluster'])  # Davies-Bouldin Index
    result_dict['HC'] = calinski_harabasz_score(X, df_labels_sectors['cluster'])  # Calinski Harabaz Index
    result_dict['Sil'] = silhouette_score(X, df_labels_sectors['cluster'])  # Silhouette Coefficient
    result_dict['hom'] = homogeneity_score(df_labels_sectors['sector'],
                                           df_labels_sectors['cluster'])  # homogenity Coefficient

    result_df = pd.Series(result_dict.values(), index=result_dict.keys())
    return result_df


def calc_metrics(port_df, df_market, riskfree_rate):
    port_df.columns = ['port']
    port_df['market'] = df_market

    result_df = pd.DataFrame()

    # Average daily returns
    mean = port_df.mean()
    result_df['AVG_returns'] = mean

    # Risk
    risk = port_df.std()
    result_df['Risk'] = risk

    # Beta
    var_ = port_df.var()
    cov_ = port_df.cov()
    beta = cov_['market'] / var_

    result_df['Beta'] = beta

    # Alpha
    alpha = mean - (riskfree_rate + beta * (result_df.loc['market', 'AVG_returns'] - riskfree_rate))
    result_df['Alpha'] = alpha

    # Sharpe
    sharpe = (mean - riskfree_rate) / risk
    result_df['Sharpe'] = sharpe

    # VaR(95%)
    VaR = - risk * 1.65
    result_df['VaR'] = VaR

    # Drawdown and Recovery
    portfolio_value = (port_df + 1).cumprod()  # датафрейм со "стоимостью" портфеля

    recovery = []
    drawdown = []
    for i in range(len(port_df.columns)):
        recovery.append(find_max_recovery(portfolio_value.iloc[:, i])[0])
        drawdown.append(find_max_drawdown(portfolio_value.iloc[:, i])[0])

    result_df['Drawdown'] = drawdown
    result_df['Recovery'] = recovery

    return result_df.T['port']


def general_pipeline(df_pct,
                     df_market,
                     df_sectors,
                     embedding_data,

                     clust_params,
                     selection_params,
                     backtesting_params,
                     ):
    """general pipeline for testing embeddings methodson financial and clustering metrics"""

    tickers_list = df_pct.columns.tolist()

    # make clustering
    norm_emb = StandardScaler().fit_transform(embedding_data)
    df_clusters = get_clusters(norm_emb, tickers_list, **clust_params)
    clust_metrics = clustering_estimation(norm_emb, df_clusters, df_sectors)

    # stock selection
    selected_tickers = select_assets(df_clusters, df_pct, **selection_params)

    df_pct_loc = df_pct.copy()
    df_pct_loc = df_pct_loc[selected_tickers]

    # port_modelling
    weights_all, return_portfolio = backtesting_one_model(df_pct_loc,
                                                          # df with pct_changes: columns - tick, index - date
                                                          **backtesting_params)

    financial_metrics = calc_metrics(return_portfolio.copy(), df_market,
                                     riskfree_rate=selection_params['riskfree_rate'])

    return weights_all, return_portfolio, clust_metrics, financial_metrics