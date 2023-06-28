from datetime import datetime
import shutil

import pandas as pd
import os
import pickle
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

from utils.portfolio import MarkowitzPortfolio

from utils.pipeline import (general_pipeline, calc_metrics, clustering_estimation, 
                            backtesting_one_model, select_assets, clustering_estimation, get_clusters)

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import warnings
warnings.filterwarnings("ignore")


def data_uploading(data_path, market_path, sectors_path, year_split, split="None"):
    df = pd.read_csv(data_path, index_col=0)

    sectors = pd.read_csv(sectors_path, index_col=0)
    sectors = sectors.sort_values('ticker')

    df_pct = df.drop(['sector'], axis=1).T
    df_pct.index = pd.to_datetime(df_pct.index)
    
    tickers_list = df_pct.columns.tolist()

    df_market = pd.read_csv(market_path, index_col=0)
    df_market.columns = ['market']
    df_market.index = pd.to_datetime(df_market.index)
    df_market = df_market.pct_change()[1:]

    if isinstance(year_split, int) and split=="train":
        df_pct = df_pct[(df_pct.index < datetime(year_split, 1, 1)) ]
        df_market = df_market[(df_market.index < datetime(year_split, 1, 1)) ]
        
    elif isinstance(year_split, int) and split=="test":
        df_pct = df_pct[(df_pct.index > datetime(year_split, 1, 1)) ]
        df_market = df_market[(df_market.index > datetime(year_split, 1, 1)) ]
    
    df_market = df_market.join(df_pct, how='inner')[['market']]      

    return df_pct, df_market, sectors, tickers_list

def general_pipeline(
    df_pct,
    df_market,
    df_sectors,
    embedding_data,
    clust_params,
    selection_params,
    backtesting_params,
):
    """
    general pipeline for testing embeddings methodson financial and clustering metrics
    """

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
    weights_all, return_portfolio = backtesting_one_model(df_pct_loc, **backtesting_params)

    financial_metrics = calc_metrics(return_portfolio.copy(), df_market,
                                     riskfree_rate=selection_params['riskfree_rate'])

    return weights_all, return_portfolio, clust_metrics, financial_metrics, df_clusters


def custom_score_clustering_gridsearch(estimator, X, y=None):
    estimator.fit(X)
    labels_predicted = estimator.labels_
    score = silhouette_score(X, labels_predicted)
    return score


def selection_sharp(list_tickers, n_save, df_pct, riskfree_rate):
    
    df_pct = df_pct[list_tickers]
    sharp = (df_pct.mean() - riskfree_rate)/df_pct.std()
    selected_tickers = sharp.sort_values(ascending=False).head(n_save).index.tolist()
    return selected_tickers

def update_config(cfg):
    riskfree_rate = (1 + cfg['riskless_rate']) **(1/252) - 1
    ret_det = (1 + cfg['ret_det']) **(1/252) - 1
    
    clust_params = dict(cfg['clust_params'])
    clust_params.update({'grid_metric':custom_score_clustering_gridsearch})
    
    clust_models_params = dict(cfg['clust_models_params'])
    
    selection_params = dict(cfg['selection_params'])
    selection_params.update({'selection_method':selection_sharp, 'riskfree_rate':riskfree_rate})
    
    backtesting_params = dict(cfg['backtesting_params'])
    backtesting_params.update({
        'port_model':MarkowitzPortfolio, 
        'ret_det':ret_det, 
        'test_start_year': cfg['year_start'] + 2, 
        'test_finish_year': cfg['year_start'] + 5, 
    })
    
    return riskfree_rate, ret_det, clust_params, clust_models_params, selection_params, backtesting_params

def modify_metrics(cluster_metrics, fin_metrics, df_clusters, emb_name, clust_model_name):
    cluster_metrics['clust_model'] = clust_model_name
    fin_metrics['clust_model'] = clust_model_name
    
    cluster_metrics['emb_model'] = emb_name
    fin_metrics['emb_model'] = emb_name
    
    df_clusters = df_clusters.set_index('ticker').rename(columns={'cluster': emb_name})    
    return cluster_metrics, fin_metrics, df_clusters

        
def calculate_benchmarks(
    df_pct,
    df_market,
    sectors,
    tickers_list,
    riskfree_rate,
    selection_params,
    backtesting_params,
    financial_metrics_df,
    cluster_metrics_df, 
    dict_port_methods, 
    dict_weight_methods,
    market_label='sp500'
):
    # market

    first_key = list(dict_port_methods.keys())[0]
    port_df = df_market.loc[dict_port_methods[first_key].index].copy()
    fin_metrics= calc_metrics(port_df, df_market, riskfree_rate=riskfree_rate)
    fin_metrics['clust_model'] = market_label
    fin_metrics['emb_model'] = market_label

    financial_metrics_df = pd.concat([financial_metrics_df, pd.DataFrame(fin_metrics).T], axis=0)  
    
    # economic sectors
    ports_df = pd.DataFrame()
    dict_weights = dict()
        
    sectors_selected = sectors.query('ticker in @tickers_list')
    clust_econ_sectors = LabelEncoder().fit_transform(sectors_selected['sector'])
    df_clusters = pd.DataFrame([tickers_list, clust_econ_sectors], index=['ticker', 'cluster']).T

    selected_tickers = select_assets(df_clusters, df_pct, **selection_params)
    df_pct_loc = df_pct[selected_tickers]
    weights_all, return_portfolio = backtesting_one_model(df_pct_loc, # df with pct_changes: columns - tick, index - date
                        **backtesting_params)
    cluster_metrics = clustering_estimation(df_pct.T.values, df_clusters, sectors)
    fin_metrics = calc_metrics(return_portfolio.copy(), df_market, riskfree_rate=riskfree_rate)

    cluster_metrics['clust_model'] = 'sectors'
    fin_metrics['clust_model'] = 'sectors'
    cluster_metrics['emb_model'] = 'sectors'
    fin_metrics['emb_model'] = 'sectors'

    cluster_metrics_df = pd.concat([cluster_metrics_df, pd.DataFrame(cluster_metrics).T], axis=0)
    financial_metrics_df = pd.concat([financial_metrics_df, pd.DataFrame(fin_metrics).T], axis=0)  

    ports_df['sectors'] = return_portfolio
    dict_weight_methods['sectors'] = weights_all

    dict_port_methods['sectors'] = ports_df
    dict_weight_methods['sectors'] = dict_weights
    
    return financial_metrics_df, cluster_metrics_df, dict_port_methods, dict_weight_methods


def save_results(
    financial_metrics_df,
    cluster_metrics_df, 
    dict_port_methods, 
    dict_weight_methods,
    financial_metric_path,
    clust_metric_path,
    port_path,
    weights_path,
):
    financial_metrics_df.to_csv(financial_metric_path)
    cluster_metrics_df.to_csv(clust_metric_path)

    with open(port_path, 'wb') as f:
        pickle.dump(dict_port_methods, f)
        
    with open(weights_path, 'wb') as f:
        pickle.dump(dict_weight_methods, f)


@hydra.main(config_path='config', config_name='validate_config', version_base=None)
def main(cfg: DictConfig):
    
    if not os.path.isdir(cfg['save_path']):
        os.mkdir(cfg['save_path'])
        
    shutil.copyfile(
        'config/embedding_config.yaml', 
        cfg['save_path'] +'/validate_config.yaml'
    )
    
    df_pct, df_market, sectors, tickers_list = data_uploading(
        cfg['data_path'], 
        cfg['market_path'], 
        cfg['sectors_path'],
        year_split=cfg['year_start'] + 2
    )

    riskfree_rate, _, clust_params, clust_models_params, selection_params, backtesting_params = update_config(cfg)
    
    dict_port_methods = dict() #Считаем портфель Марковица для всех методов кластеризации
    dict_weight_methods = dict()
    cluster_metrics_df = pd.DataFrame() 
    financial_metrics_df = pd.DataFrame() 
    clusters_df = pd.DataFrame()

    all_files = os.listdir(cfg['embedding_path'])
    for emb_path in tqdm(all_files):
        
        emb_name = emb_path.replace('.csv', '')   
        print(emb_name)     
        embeddings = pd.read_csv(cfg['embedding_path'] + '/' + emb_path, index_col=0).loc[tickers_list].values
        
        ports_df = pd.DataFrame()
        dict_weights = dict()
        
        for clust_model_name, clust_model in cfg['clust_models'].items():
            
            clust_model = instantiate(clust_model)
            clust_params.update({'clust_model': clust_model, 'grid_params': clust_models_params[clust_model_name]})
            weights_all, return_portfolio, cluster_metrics, fin_metrics, df_clusters = general_pipeline(
                df_pct,
                df_market,
                sectors, 
                embedding_data=embeddings,
                clust_params=clust_params,
                selection_params=selection_params,
                backtesting_params=backtesting_params
            )
            
            cluster_metrics, fin_metrics, df_clusters = modify_metrics(cluster_metrics, fin_metrics, df_clusters, emb_name, clust_model_name)
            
            cluster_metrics_df = pd.concat([cluster_metrics_df, pd.DataFrame(cluster_metrics).T], axis=0)
            financial_metrics_df = pd.concat([financial_metrics_df, pd.DataFrame(fin_metrics).T], axis=0)
            clusters_df = pd.concat([clusters_df, df_clusters], axis=1)
            ports_df[clust_model_name] = return_portfolio
            dict_weights[clust_model_name] = weights_all
            
        dict_port_methods[emb_name] = ports_df
        dict_weight_methods[emb_name] = dict_weights
        
    financial_metrics_df, cluster_metrics_df, dict_port_methods, dict_weight_methods = calculate_benchmarks(
        df_pct,
        df_market,
        sectors,
        tickers_list,
        riskfree_rate,
        selection_params,
        backtesting_params,
        financial_metrics_df,
        cluster_metrics_df, 
        dict_port_methods, 
        dict_weight_methods,
    )
    
    save_results(
        financial_metrics_df,
        cluster_metrics_df, 
        dict_port_methods, 
        dict_weight_methods,
        cfg['financial_metric_path'],
        cfg['clust_metric_path'],
        cfg['port_path'],
        cfg['weights_path'],
    )
    
if __name__ == '__main__':
    main()