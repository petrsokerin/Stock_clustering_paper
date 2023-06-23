import datetime
import pandas as pd

def data_uploading(cfg):
    df = pd.read_csv(cfg['data_folder'], index_col=0)

    sectors = df['sector']

    df_pct = df.drop(['sector'], axis=1).T
    df_pct.index = pd.to_datetime(df_pct.index)

    tickers_list = df_pct.columns.tolist()

    df_market = pd.read_csv(cfg['ticker_data_sp500'], index_col=0)
    df_market.columns = ['market']
    df_market.index = pd.to_datetime(df_market.index)
    df_market = df_market.pct_change()[1:]

    
    if isinstance(cfg['year_split'], int):
        df_pct = df_pct[(df_pct.index < datetime(cfg['year_split'], 1, 1)) ]
        df_market = df_market[(df_market.index < datetime(cfg['year_split'], 1, 1)) ]
    else:
        df_market = df_market.join(df_pct, how='inner')[['market']]      

    print(len(sectors))
    return df_pct, df_market, sectors