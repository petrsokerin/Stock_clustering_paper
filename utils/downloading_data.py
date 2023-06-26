import os
import shutil

import pandas as pd
from pandas_datareader import data as web
from tqdm import tqdm

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import yfinance as yf
from finvizfinance.quote import finvizfinance



def loading_dataset(tickers, start_date, end_date, meaning):

    yf.pdr_override()

    df = pd.DataFrame([])
    for tick in tqdm(tickers):
        try:
            panel_data = web.get_data_yahoo(tick, start_date, end_date)
            temp = pd.DataFrame(data={tick: panel_data[meaning]}, index=panel_data.index)
            df = pd.concat([df, temp], axis=1)

        except Exception as e:
            print(e)
            print('Ticker', tick, 'failed')

    return df


def generating_datasets(
    path_df_tickers, 
    start_date, 
    end_date, 
    #sector_path, 
    meaning='Close'
):
    tickers_df = pd.read_csv(path_df_tickers, sep=';')
    tickers = tickers_df['Ticker'].values.tolist()

    # df_sectors = pd.read_csv(sector_path, index_col=0)
    # set_tickers_from_sect = set(df_sectors['ticker'].values.tolist())
    # tickers = list(set(tickers) & set_tickers_from_sect)
    
    tickers = [tick for tick in tickers if '.' not in tick]

    df_stock_price = loading_dataset(tickers, start_date, end_date, meaning)

    # set_tickers_from_market = set(df_stock_price.columns.tolist())
    # tickers_to_save = list(set_tickers_from_sect & set_tickers_from_market)
    # df_stock_price = df_stock_price[tickers_to_save]
    return df_stock_price  

def getting_sect_per_ticker(tickers):
    dict_tick_sect = dict()
    for ticker in tqdm(tickers):
        try:
            stock = finvizfinance(ticker)
            dict_tick_sect[ticker] = stock.ticker_fundament()['Sector']
        except:
            print(ticker)
            continue
    return dict_tick_sect


def generate_sector_datasets(path_df_tickers):
    
    tickers_df = pd.read_csv(path_df_tickers, sep=';')
    tickers = tickers_df['Ticker'].values.tolist()
    dict_tick_sect = getting_sect_per_ticker(tickers)
    df_sectors = pd.DataFrame(data={'ticker': dict_tick_sect.keys(), 'sector': dict_tick_sect.values()})
    return df_sectors


def preprocessed(df_stock_data, path_sectors, thr_rate=0.05):

    df_na = df_stock_data.isna().sum()
    threshold = thr_rate * len(df_stock_data)
    stocks_to_drop = df_na[df_na > threshold].index.tolist()
    df_stock_data = df_stock_data.drop(stocks_to_drop, axis=1)
    df_stock_data = df_stock_data.dropna(axis=0)

    df_stock_data = df_stock_data.pct_change()[1:]
    df_stock_data = df_stock_data.T

    df_sectors = pd.read_csv(path_sectors, index_col=0)
    dict_tick_sect = dict(zip(df_sectors['ticker'].values.tolist(),
                              df_sectors['sector'].values.tolist()))

    set_tickers_from_sect = set(df_sectors['ticker'].values.tolist())
    set_tickers_from_close = set(df_stock_data.index.tolist())
    tickers_to_save = list(set_tickers_from_sect & set_tickers_from_close)
    df_stock_data = df_stock_data.loc[sorted(tickers_to_save)]

    df_stock_data['sector'] = df_stock_data.index.map(dict_tick_sect)
    return df_stock_data
    

@hydra.main(config_path='../config', config_name='download_config', version_base=None)
def main(cfg: DictConfig):
    
    path = '../'
    
    if not os.path.isdir(path + cfg['save_folder']):
        os.makedirs(path + cfg['save_folder'])
        shutil.copyfile(path + 'config/download_config.yaml', path + cfg['save_folder'])
    
    # with open(path + 'config/config.json', 'r') as file:
    #     config = json.load(file)

    start_date, end_date = cfg['start_date'], cfg['end_date']
    path_df_tickers = path+cfg['ticker_companies']
    
    # benchmark downloading 
    if cfg['download_market']:
        df_market = loading_dataset(
            [cfg['market_ticker']], 
            start_date, 
            end_date, 
            'Close'
        )
        df_market.to_csv(path+cfg['ticker_data_market'])
    
    # market data downloading
    if cfg['download_market']:
        if cfg['download_meaning'] == 'Close':
            save_stock_data_path = path+cfg['ticker_data_close']
        elif cfg['download_meaning'] == 'Volume':
            save_stock_data_path = path+cfg['ticker_data_volume']
        else:
            raise ValueError('Download meaning should be Close or Volume')
        
        df_stock_data = generating_datasets(
            path_df_tickers, 
            start_date, 
            end_date, 
            #sector_path=path+cfg['ticker_sectors_path'], 
            meaning=cfg['download_meaning'],
        )
        df_stock_data.to_csv(save_stock_data_path)
    else:
        df_stock_data = pd.read_csv(path+cfg['ticker_data_close'], index_col=0)
        
    # sectors downloading
    if cfg['download_sectors']:
        df_sectors = generate_sector_datasets(path_df_tickers)
        df_sectors.to_csv(path+cfg['ticker_sectors_path'])
        
    # data preprocessing
    if cfg['preprocessed']:
        df_stock_prepr = preprocessed(
            df_stock_data, 
            path+cfg['ticker_sectors_path'], 
            
        )
        df_stock_prepr.to_csv(path+cfg['ticker_data_preprocessed'])    

if __name__ == "__main__":
    main()

