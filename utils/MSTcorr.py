from sklearn.cluster import AgglomerativeClustering
import numpy as np

class MSTcorrClustering:

    def __init__(self, n_clusters, linkage='complete'):
        self.n_clusters = n_clusters
        self.linkage = linkage
                
        self.clust_model = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='precomputed',
                                       linkage=self.linkage)

    def fit(self, data):
        corr = np.corrcoef(data)
        D = 1 - corr ** 2
        self.clust_model.fit(D)
        self.labels_ = self.clust_model.labels_
        
        return self
        
        
    def get_params(self):
        return self.clust_model.get_params()
    
    def set_params(self, **params):
        all_params = self.clust_model.get_params()
        all_params.update(params)
        self.clust_model.set_params(**all_params)
        return self