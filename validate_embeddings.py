from datetime import datetime
from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import json
import pickle
from tqdm.notebook import tqdm

from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN)

from sklearn.metrics import silhouette_score

from utils.portfolio import MarkowitzPortfolio
from utils.portfolio_metrics import (calculate_measures, show_drawdown_recovery, 
                                     find_max_recovery, find_max_drawdown)

from utils.pipeline import (general_pipeline, calc_metrics, clustering_estimation, 
                            backtesting_one_model, select_assets, clustering_estimation, get_clusters)

import warnings
warnings.filterwarnings("ignore")