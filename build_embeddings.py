from datetime import datetime
from typing import Any
from math import floor
import json
import os

import pandas as pd
import numpy as np
import torch

from tqdm.notebook import tqdm

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils.table_feature_calculation import financial_feature_calculation
from utils.load_datasets import data_uploading


@hydra.main(config_path='config', config_name='embedding_config', version_base=None)
def main(cfg: DictConfig):
    df_pct, df_market, sectors= data_uploading(cfg)

    if not os.path.isdir(cfg['save_path']):
        os.mkdir(cfg['save_path'])

    for method in cfg['embedding_methods']:
        print(method)
        method_inst = instantiate(method)
        #print(type(method))
        method_name, method_inst = next(iter(method_inst.items()))
        #print(method_inst.save_path)
        method_inst.generate_embeddings(data=df_pct, market_data=df_market, sectors=sectors)
        method_inst.save_embeddings()

if __name__ == "__main__":
    main()