# stock_clustering

Repo for final project of Sequential data modelling course in My University '22.

## Requirements

  - pandas==1.3.5
  - pandas-datareader==0.10.0
  - finvizfinance==0.12.2 *
  - scipy==1.7.1
  - numpy==1.19.1
  - matplotlib==3.3.1
  - seaborn==0.11.2
  - plotly==5.3.1
  - sklearn==1.0.2
  - gensim==3.6.0
  - giotto-tda==0.6.0
  - pytorch-lightning==1.5.10
  - torch==1.10.2

All the requirements are listed in `requirements.txt`

For install all packages run 

```
pip install -r requirements.txt
```
## Describtion

The main idea of the project is to compare different time-series embedding extraction methods for stock clustering to reduce risks of investments.
We used S&P 500 stock prices from dates 2018-2022 to cluster them alternatively to economical sectors, and performed portfolio optimisation task in terms of risk minimisation.
The repository provides data and code to repeat our calculations, as well as results in a corresponding folder.

## Content


| File or Folder | Content |
| --- | --- |
| advance_embeddings| folders contains jupyter notebooks with topological approach, signal2vec model and transformer model |
| autoencoders | folders contains jyputer notebook for training autoencoder models and using them for getting embeddings|
| config | folder contains config files with params of models and paths |
| data | folder contains datasets loaded with yahoo finance and folders with embeddings from different algorithms |
| ts2vec | folder is a fork from ts2vec model. Original [TS2Vec repo](https://github.com/yuezhihan/ts2vec)  |
| utils | folder with necessary functions |
| Pipeline.ipynb | jupyter notebook contains pipeline for experiments with different embeddings|
| primary_embeddings.ipynb | jupyter notebook contains calculation all embeddings excluding topological approaches, transformers approaches and signal2vec model |

## Results

As a result we succeed in providing less risky method of stock selection in comparison with economical sectors.
Topological approaches for time series clustering perform the best according to as financial as clustering metrics. 
Economic sectors don't map with clusters from different methods at all. 

## Contacts

| **Name** | **Telegram** |
|----:|:----------:|
| Petr Sokerin | @Petr_Sokerin |
| Elizaveta Makhneva | @eliza_1 |
| Kristian Kuznetsov | @pyashy |
