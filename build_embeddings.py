import os
import shutil

import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils.table_feature_calculation import financial_feature_calculation


def data_uploading(data_path, market_path, year_split, split="train"):
    df = pd.read_csv(data_path, index_col=0)

    sectors = df['sector']

    df_pct = df.drop(['sector'], axis=1).T
    df_pct.index = pd.to_datetime(df_pct.index)

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

    return df_pct, df_market, sectors


@hydra.main(config_path='config', config_name='embedding_config', version_base=None)
def main(cfg: DictConfig):
    
    df_pct, df_market, sectors= data_uploading(cfg['data_path'], cfg['market_path'], cfg['year_split'])

    if not os.path.isdir(cfg['save_path']):
        os.mkdir(cfg['save_path'])
        
    shutil.copyfile(
        'config/embedding_config.yaml', 
        cfg['save_folder'] +'/embedding_config.yaml'
    )

    for method in tqdm(cfg['embedding_methods']):
        method_inst = instantiate(method)
        #print(type(method))
        _, method_inst = next(iter(method_inst.items()))
        method_inst.generate_embeddings(data=df_pct, market_data=df_market, sectors=sectors)
        method_inst.save_embeddings()

if __name__ == "__main__":
    main()