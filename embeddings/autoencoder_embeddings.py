from .base_embedding import Embedding
import pandas as pd

class Autoencoder_embedding(Embedding):
    def __init__(self, emb_model, emb_params):
        super().__init__()
        self.emdedding_model = emb_model(**emb_params)
        self.embeddings = pd.DataFrame([])

    def generate_embeddings(self, data, tickers_list):
        embeddings = self.emdedding_model.fit_transform(data)
        self.embeddings = pd.DataFrame(embeddings, index=tickers_list)
        return self.embeddings
