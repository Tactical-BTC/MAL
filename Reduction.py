import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,minmax_scale, MaxAbsScaler, RobustScaler, Normalizer, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from abc import *


def data_grouping(path):
    '''
    path 지정해주면 동작함
    '''
    list = os.listdir(path)
    list_path = [os.path.join(path, i) for i in list]
    list_df = [pd.read_csv(i, index_col=0) for i in list_path]
    df = [dataframe.groupby(['labels','Column','Field']).median() for dataframe in list_df]
    df_reset = [dataframe.reset_index(level=0) for dataframe in df]
    df_trim = [dataframe.iloc[:,2:] for dataframe in df_reset]
    
    return df_reset, df_trim 


class Reduction:
    
    def __init__(self, path, scaler):
        self.df_reset, self.df_list = data_grouping(path=path)
        self.scaler = scaler()
    
    def pca_fit(self):
        scaler = self.scaler
        X_scaled = [scaler.fit_transform(i) for i in self.df_list]
        pca=PCA(n_components=2)
        X_pca = [pca.fit_transform(i) for i in X_scaled]
        X_pca_df = [pd.DataFrame(i) for i in X_pca]
        label = [self.df_reset[i]['labels'].values for i in range(len(X_pca_df))]
        B = [pd.DataFrame(i, columns=['labels']) for i in label]
        
        X_pca = []
        for i in range(len(X_pca_df)):
            X_pca_df[i].loc[:,'Stages'] = B[i]
            X_pca_df[i].columns = ['Component 1','Component 2','Stages']
            X_pca.append(X_pca_df[i])
        
        return X_pca

    def tsne_fit(self):
        scaler = self.scaler
        X_scaled = [scaler.fit_transform(i) for i in self.df_list]
        tSNE=TSNE(n_components=2)
        X_tsne = [tSNE.fit_transform(i) for i in X_scaled]
        X_tsne_df = [pd.DataFrame(i) for i in X_tsne]
        label = [self.df_reset[i]['labels'].values for i in range(len(X_tsne_df))]
        B = [pd.DataFrame(i, columns=['labels']) for i in label]
        
        X_tsne = []
        for i in range(len(X_tsne_df)):
            X_tsne_df[i].loc[:,'Stages'] = B[i]
            X_tsne_df[i].columns = ['Component 1','Component 2','Stages']
            X_tsne.append(X_tsne_df[i])
    
        return X_tsne
        
        
    def umap_fit(self):
        scaler = self.scaler
        X_scaled = [scaler.fit_transform(i) for i in self.df_list]
        reducer = umap.UMAP()
        X_umap = [reducer.fit_transform(i) for i in X_scaled]
        X_umap_df = [pd.DataFrame(i) for i in X_umap]
        label = [self.df_reset[i]['labels'].values for i in range(len(X_umap_df))]
        B = [pd.DataFrame(i, columns=['labels']) for i in label]
        
        X_umap = []
        for i in range(len(X_umap_df)):
            X_umap_df[i].loc[:,'Stages'] = B[i]
            X_umap_df[i].columns = ['Component 1','Component 2','Stages']
            X_umap.append(X_umap_df[i])
            
        return X_umap
        

def visualization(A):

    fig = px.scatter(A, x='Component 1', y='Component 2',
                        color='Stages',height=800, width=1000,
                        template = 'plotly_white')
    fig.show()
    
        
        