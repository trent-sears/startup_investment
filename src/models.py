import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 17})
import folium
import pickle
import re
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

class DataFrame(object):

    def __init__(self,file_path):
        """Create df object
        Parameters
        ----------
        file_path: takes in a file path to raw data
        """
        self.file_path = file_path
        
    def clean(self):
        '''
        Parameters
            self: self
        Returns
            a cleaned df
        '''
        df = pd.read_csv(self.file_path,encoding='latin1')
        df.dropna(inplace=True)
        df.drop(['permalink','region','founded_month','founded_quarter'],axis=1,inplace=True)
        df['founded_at'] = pd.to_datetime(df['founded_at'],errors='coerce')
        df['first_funding_at']= pd.to_datetime(df['first_funding_at'],errors='coerce')
        df['last_funding_at']= pd.to_datetime(df['first_funding_at'],errors='coerce')
        df['founded_year'] = df['founded_year'].astype('int64')
        df.drop(df[df['country_code']=='CAN'].index,inplace=True)
        df['funding_total_usd'] = df[' funding_total_usd '].apply(lambda x: x.replace(' ',''))\
            .apply(lambda x: x.replace(',',''))
        df['funding_total_usd'] = df['funding_total_usd'].apply(lambda x: x.replace('-','0'))
        df['funding_total_usd'] = df['funding_total_usd'].astype('int64')
        df['market'] = df[' market '].apply(lambda x: x.replace(' ','')) 
        df.drop([' market ',' funding_total_usd ','country_code','homepage_url','name','city','last_funding_at',
            'round_A', 'round_B','round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H','category_list'],\
            axis=1,inplace=True)
        return df

def feature_engineer(df):
    '''
    Parameters
        df: Takes in a pandas data frame
    Returns
        a data frame with engineered features
    '''
    df['time_to_funding'] = abs((df['first_funding_at']-df['founded_at']).dt.days)
    test_list = list(df['market'].value_counts()\
        .rename_axis('market').reset_index(name='counts')[:20]['market'])
    df.loc[~df["market"].isin(test_list), "market"] = "Other"
    df.dropna(inplace=True)
    return df

def add_finance(symbol,df):
    '''
    Parameters
        symbol: stock ticker
        df: data frame to add 
    Returns
        a data frame with engineered features
    '''
    finance = yf.Ticker(symbol)
    fin_df = finance.history(period="max")
    fin_df['avg']=(fin_df['Open']+fin_df['Close'])/2
    fin_df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'],axis=1,inplace=True)
    fin_df['first_funding_at'] = fin_df.index
    
    no_finance_data = list(clean_feat_df[clean_feat_df['first_funding_at']<fin_df['first_funding_at'].min()].index)
    df.drop(no_finance_data,inplace=True)
    df.set_index('first_funding_at')
    df = df.merge(fin_df,on='first_funding_at')
    return df

def eval_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print('===ACCURACY===')
    print(acc)
    print('===RECALL===')
    print(rec)
    print('===PRECISION===')
    print(prec)
    print('===F1===')
    print(f1)

if __name__ == '__main__':

    intial_df = DataFrame('../../../Downloads/investments_VC.csv').clean()
    clean_feat_df=feature_engineer(intial_df)
    clean_feat_df=add_finance('^VIX',clean_feat_df)

    #change funding from to dollars to ones and zeros
    funding_type_df = clean_feat_df.loc[:,'seed':'product_crowdfunding']\
        .apply(lambda x: x>0).astype('int64')
    funding_type_df.drop('undisclosed',axis=1,inplace=True)

    #make dummies
    col_list = ['Advertising', 'Analytics', 'Biotechnology', 'CleanTechnology',
    'Consulting', 'CuratedWeb', 'E-Commerce', 'Education',
    'EnterpriseSoftware', 'Finance', 'Games', 'Hardware+Software',
    'HealthCare', 'HealthandWellness', 'Manufacturing', 'Mobile',
    'Security', 'Semiconductors', 'SocialMedia', 'Software','Other']
    market_dummies = pd.get_dummies(clean_feat_df['market']).reindex(columns=col_list)
    state_dummies = pd.get_dummies(clean_feat_df['state_code'])

    #set targets as ones and zeros
    clean_feat_df['status'] = clean_feat_df['status'].apply(lambda x: x.replace('operating','0'))\
        .apply(lambda x: x.replace('acquired','1')).apply(lambda x: x.replace('closed','0'))
    clean_feat_df['status'] = clean_feat_df['status'].astype('int64')

    #set X and Y, test train split and SMOTE                           
    X =market_dummies.iloc[:,:20].join(state_dummies.iloc[:,:50])\
        .join(clean_feat_df['time_to_funding']).join(clean_feat_df['avg']).join(funding_type_df).values
    y=clean_feat_df['status'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    #build and pickle logisitic model
    log_model = LogisticRegression(solver="lbfgs",max_iter=300)
    log_model.fit(X_train, y_train)
    y_predict_log = log_model.predict(X_test)

    #build and pickle random forest model
    rf = RandomForestClassifier(max_features=42, n_estimators=100)
    rf.fit(X_train, y_train)
    y_predict_rf = rf.predict(X_test)

    #build and pickle extra trees model
    extra = ExtraTreesClassifier(n_estimators=100)
    extra.fit(X_train, y_train)
    y_predict_extra =extra.predict(X_test)

    #build and pickle gradient boosted model
    gradient= GradientBoostingClassifier(random_state=1)
    gradient.fit(X_train, y_train)
    y_predict_gradient = gradient.predict(X_test)
                                 
    #Make feature importance plot on random forest
    X =market_dummies.iloc[:,:20].join(state_dummies.iloc[:,:50])\
        .join(clean_feat_df['time_to_funding']).join(funding_type_df)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]
    fig,ax = plt.subplots(figsize=(14,7))
    ax.bar(range(0,10),importances[indices[:10]],color="r", yerr=std[indices[:10]], align="center")
    plt.xticks(range(0,10), ['Venture', 'Time to Funding','CA','Seed','Debt Financing', 'MA',\
     'Software','Biotechnology','NY','Enterprise Software'],rotation=30,ha='right')
    ax.set_xlabel('Features')
    ax.set_ylabel('Feature Weights')
    ax.set_title('Feature Importances',fontweight='bold')
    plt.tight_layout()
    plt.savefig('../images/feature_importances1.png')


