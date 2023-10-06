import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,minmax_scale, MaxAbsScaler, RobustScaler, Normalizer, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import seaborn as sns
import os
import plotly.express as px


from abc import *


class Remove_corr:
    '''
    상관관계 0.9이상 제거
    '''
    def __init__(self, df):
        self.df = df
    
    def return_df(self, cut_value=0.9):
        '''
        cut_value를 통해서 조절 default: 0.9
        '''
        corr_df = self.df.corr().abs()
        # Create and apply mask
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        tri_df = corr_df.mask(mask)

        # Find index of columns with correlation greater than 0.9
        to_drop = [column for column in tri_df.columns if any(tri_df[column] > cut_value)]
        reduced_df = self.df.drop(to_drop, axis=1)
        
        return reduced_df

    
    

class PCA_fit:
    '''
    sckit-learn에서 standard, minmax, etc..
    '''
    def __init__(self, scaler,df):
        self.scaler = scaler()
        self.df = df
    
    def normalization(self):
        scaler = self.scaler
        scaler.fit(self.df)
        X_scaled = scaler.transform(self.df)
        pca=PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        return X_pca
    
    def visulaization(self):
        pass
        # plt.figure(figsize=(12,8))
        # plt.scatter(*X_pca.T, linewidth=0, s=50, c='b', alpha=0.25);


class Outlier:
    
    @abstractclassmethod
    def __init__(self):
        pass
    
    @abstractclassmethod
    def return_df(self):
        pass
    

class MCD_outlier(Outlier):
    
    def __init__(self,x_pca):
        self.x_pca = x_pca
            
    def return_df(self):
        robust_cov = MinCovDet().fit(self.x_pca)
        AA = robust_cov.mahalanobis(self.x_pca)[:,np.newaxis]
        X_pca_df = pd.DataFrame(self.x_pca, columns = ['X0','X1'])
        X_pca_df['Mahalanobis'] = AA
        X_pca_df['p'] = 1 - chi2.cdf(X_pca_df['Mahalanobis'],1)
        X_pca_df.loc[X_pca_df.p < 0.001, 'labels'] = 'outliers'
        X_pca_df.loc[X_pca_df.p >= 0.001, 'labels'] = 'inners'
        X_pca_df.columns =['PC1','PC2','Mahalanobis','p','labels']
        
        return X_pca_df
    
      
class IsoForest_outlier(Outlier):

    def __init__(self,x_pca):
        self.x_pca = x_pca
            
    def return_df(self):
        clf=IsolationForest(max_samples='auto', contamination='auto', 
                        max_features=1.0, bootstrap=True)
        clf.fit(self.x_pca)
        pred = clf.predict(self.x_pca)
        A = pd.DataFrame(self.x_pca)
        A['labels'] = pred
        A.loc[A['labels']== 1, 'labels'] = 'inners'
        A.loc[A['labels']== -1, 'labels'] = 'outliers'
        A.columns = ['PC1','PC2','labels']
                        
        return A
    


class OneClassSVM_outlier(Outlier):
    
    def __init__(self,x_pca):
        self.x_pca = x_pca
            
    def return_df(self, gamma='auto',nu=0.03):
        svm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        svm.fit(self.x_pca)
        pred = svm.predict(self.x_pca)
        A = pd.DataFrame(self.x_pca)
        A['labels'] = pred
        A.loc[A['labels']== 1, 'labels'] = 'inners'
        A.loc[A['labels']== -1, 'labels'] = 'outliers'
        A.columns = ['PC1','PC2','labels']
                        
        return A


class Visulalization:
    
    def __init__(self, df):
        self.df = df

    def visualize(self):
        pallete = sns.color_palette('Paired')
        fig = sns.scatterplot(data=self.df, x='PC1',y='PC2',hue='labels', 
                              palette={'inners':pallete[1], 'outliers':pallete[0]})
        # 그림 그리는 코드수정
        
        return fig 


class Coordinater:
    '''
    편의상 OneClassSVM만 사용하였음.
    fig(inner & outlier 표시),
    Final(df): outliers가 제거된 데이터
    '''
    def __init__(self,scaler,df):
        self.scaler = scaler
        self.df = df
            
    def final_results(self):
        reduced_df = Remove_corr(self.df)
        x_pca = PCA_fit(self.scaler, reduced_df).normalization()
        oneclasssvm = OneClassSVM_outlier(x_pca).return_df()
        Final = reduced_df.return_df().loc[oneclasssvm.query('labels == "inners"').index,:] 
        #inner만 Final에 넣음, outlier제거
        
        return Final
    
    def visualize(self):
        reduced_df = Remove_corr(self.df)
        x_pca = PCA_fit(self.scaler, reduced_df).normalization()
        oneclasssvm = OneClassSVM_outlier(x_pca).return_df()
        fig = Visulalization(oneclasssvm).visualize()
        
        return fig
    
    
def save(df):
    '''
    전처리 후 processed_data에 저장하는 함수
    '''    
    for i in range(len(df)):
        path = r'C:\Users\jihoon.park\Desktop\Paper preparation\Processed_data'
        name = 'processed_df' + '{}'.format(i) + '.csv'
        filepath = os.path.join(path, name)
        df[i].to_csv(filepath)
        
        
        
        


        
def Preprocessed_visual(path):
    dir = r'C:\Users\jihoon.park\Desktop\Paper preparation\SimA_20230214\CSV file'
    os.chdir(os.path.join(dir,path))
    filelist = os.listdir()
    filepath = [os.path.join(os.getcwd(),i) for i in filelist]
    df = [pd.read_csv(filepath[i]) for i in range(0,len(filepath))]
    df2 = [df[i].drop(columns = ['MITO_count', 'Nucleus_count']) for i in range(0,len(filepath))]
    df_val = [df2[i].loc[:,'Nucleus_Intensity Mean':'RBC_Texture SER Dark 0 px'] for i in range(0,len(filepath))]
    Final = [Coordinater(StandardScaler,df_val[i]).final_results() for i in range(len(df_val))]
    return df, Final



    
def Removed_df(df, i):
    B = df[i].iloc[:,5:-1]
    B_removed = Remove_corr(B).return_df()
    return B_removed
    
def Removed_df_save(df, i):   
    Alphabet = ['A','B','C','D','E','F','G','H']
    A = df[i].iloc[:,:5] 
    B_removed = Removed_df(df, i)
    C = df[i].iloc[:,[-1]]
    D = pd.concat([A,B_removed,C], axis=1)
    D.to_csv('Corr_removed' + '_' + Alphabet[i] + '.csv')
    