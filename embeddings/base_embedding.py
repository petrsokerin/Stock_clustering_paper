from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch

from tsfresh import extract_features
from sklearn.feature_selection import SelectKBest
from utils.table_feature_calculation import financial_feature_calculation
from autoencoders.autoencoders import Conv1dAutoEncoder, LSTMAutoEncoder, MLPAutoEncoder


class Embedding(ABC):
    def __init__(self, path, **kwargs):
        self.embeddings = pd.DataFrame([])
        self.save_path = path

    @abstractmethod
    def generate_embeddings(self, data, *args, **kwargs):
        pass
    
    def save_embeddings(self):
        self.embeddings.to_csv(self.save_path)



class ClassicEmbedding(Embedding):
    def __init__(self, emb_model, **kwargs):
        super().__init__(**kwargs)
        self.emdedding_model = emb_model
        self.embeddings = pd.DataFrame([])

    def generate_embeddings(self, data, *args, **kwargs):
        tickers_list = data.columns

        embeddings = self.emdedding_model.fit_transform(data.T)
        self.embeddings = pd.DataFrame(embeddings, index=tickers_list)
        return self.embeddings


class AutoencoderEmbedding(Embedding):
    def __init__(self, model_name, model_params, **kwargs):
        super().__init__(**kwargs)
        
        if model_name == 'mlp':
            self.emb_model = MLPAutoEncoder.load_from_checkpoint(**model_params)
        elif model_name == 'cnn':
            self.emb_model = Conv1dAutoEncoder.load_from_checkpoint(**model_params)
        elif model_name == 'lstm':
            self.emb_model = LSTMAutoEncoder.load_from_checkpoint(**model_params)
        else:
            raise ValueError('Incorrect model name. Should be lstm, mlp or cnn.')
        self.emb_model.eval()


    def generate_embeddings(self, data, *args, **kwargs):
        tickers_list = data.columns
        embeddings = np.zeros((len(tickers_list), 100))

        for i, name_ticker in enumerate(tickers_list):

            ts_name = data[name_ticker].values
            ts_name = ts_name.flatten()

            seq_len = ts_name.shape[0]

            floor_ = seq_len // 100
            sample = ts_name[:100 * floor_].reshape(floor_, 1, 100)
            
            global_embeddings = self.emb_model.predict_step(torch.tensor(sample).float())
            if len(global_embeddings.shape) > 2:
                global_embeddings.squeeze()

            pooled_embedding = global_embeddings.detach().numpy().mean(axis=0)           
            embeddings[i, :] = pooled_embedding
        self.embeddings = pd.DataFrame(embeddings, index=tickers_list)



class FinancialMetricEmbedding(Embedding):
    def __init__(self, riskless_rate, **kwargs):
        super().__init__(**kwargs)
        self.riskless_rate = riskless_rate

    def generate_embeddings(self, data, market_data, *args, **kwargs):
        self.embeddings = financial_feature_calculation(data.T, market_data.T,  self.riskless_rate)
        return self.embeddings
    

class TSFreshEmbedding(Embedding):
    def __init__(self, n_features, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features

    def generate_embeddings(self, data, sectors, *args, **kwargs):

        tickers_list = data.columns

        df_to_tsfresh = data.reset_index()
        df_to_tsfresh = pd.melt(df_to_tsfresh, id_vars=['index'], var_name='ticker')

        data_tsfresh = extract_features(df_to_tsfresh, column_id='ticker', n_jobs=4, column_sort='index')
        data_tsfresh = data_tsfresh.dropna(axis=1)
        print(data_tsfresh.shape, len(sectors))
        features_filtered = SelectKBest(k=self.n_features).fit_transform(data_tsfresh, sectors)

        self.embeddings = pd.DataFrame(features_filtered, index=tickers_list)


class TS2VecEmbedding(Embedding):
    def __init__(self, emb_model, **kwargs):
        super().__init__(**kwargs)
        self.emb_model = emb_model


    def generate_embeddings(self, data, *args, **kwargs):

        tickers_list = data.columns

        data = np.expand_dims(data.values.T, axis=2)

        # Train a TS2Vec model
        self.emb_model.fit(data, verbose=False)

        emb_ts2vec = self.emb_model.encode(data, encoding_window='full_series') 
        self.embeddings = pd.DataFrame(emb_ts2vec, index=tickers_list)

