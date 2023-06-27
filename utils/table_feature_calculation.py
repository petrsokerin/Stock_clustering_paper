from .feature_functions import find_max_recovery, find_max_drawdown
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# path = '../'

# with open(path+'config/config.json', 'r') as file:
#     config = json.load(file)
    

def financial_feature_calculation(df_no_sector, df_with_market, riskless_rate=0.03/252):
    
    table_features = pd.DataFrame(index=df_no_sector.index)
    
    table_features['mean_return'] = df_no_sector.T.mean()
    table_features['std_return'] = df_no_sector.T.std()
    table_features['median_return'] = df_no_sector.T.median()
    table_features['share_positive_return'] = (df_no_sector.T > 0).sum() / df_no_sector.shape[1]

    features_names = ['max_drawdown', 'rec_period', 'beta', 'alpha',
                     'sharp', 'VaR', 'CVaR', 'CAPM', 'coef_var', 'IR']
    dict_features = {name:[] for name in features_names}

    index = df_with_market.loc['market'].T.values
    r_market = np.mean(index)

    for ticker in tqdm(df_no_sector.index):
        price = df_no_sector.loc[ticker].T.values
        price_cumprod = (df_no_sector.loc[ticker] + 1).cumprod()

        max_rec_per = find_max_recovery(price_cumprod)[0]
        max_drawdown = find_max_drawdown(price_cumprod)[0]

        covar = np.cov(price, index)[0, 1]
        std = table_features.loc[ticker, 'std_return']
        var = std ** 2
        var_market = np.var(index)
        mean_return = table_features.loc[ticker, 'mean_return']

        beta = covar / var_market
        alpha = mean_return - beta * r_market
        sharp = (mean_return - riskless_rate) / std
        VaR = np.quantile(price, 0.05)
        CVaR = price[price < VaR].mean()
        CAPM = riskless_rate + beta * (r_market - riskless_rate)
        coef_variation = var / mean_return
        IR = (mean_return - r_market) / np.std(price - index)

        feature_meanings = [max_drawdown, max_rec_per, beta, alpha, sharp,
                            VaR, CVaR, CAPM, coef_variation, IR]
        dict_feature_meanings = dict(zip(features_names, feature_meanings))
        for name, meaning in dict_feature_meanings.items():
            dict_features[name].append(meaning)

    for name, column in dict_features.items():
        table_features[name] = column
    
    return table_features

