import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17})
import folium

from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def clean(file_path):
    '''
    Parameters
        file_path: takes in a file path
    Returns
        a cleaned df
    '''
    df = pd.read_csv(file_path,encoding='latin1')
    df.dropna(inplace=True)
    df.drop(['permalink','region','founded_month','founded_quarter'],axis=1,inplace=True)
    df['founded_at'] = pd.to_datetime(df['founded_at'],errors='coerce')
    df['first_funding_at']= pd.to_datetime(df['first_funding_at'],errors='coerce')
    df['last_funding_at']= pd.to_datetime(df['first_funding_at'],errors='coerce')
    df['founded_year'] = df['founded_year'].astype('int64')
    df.drop(df[df['country_code']=='CAN'].index,inplace=True)
    df['funding_total_usd'] = df[' funding_total_usd '].apply(lambda x: x.replace(' ','')).apply(lambda x: x.replace(',',''))
    df['funding_total_usd'] = df['funding_total_usd'].apply(lambda x: x.replace('-','0'))
    df['funding_total_usd'] = df['funding_total_usd'].astype('int64')
    df['market'] = df[' market '].apply(lambda x: x.replace(' ',''))
    df.drop(' market ',axis=1,inplace=True)
    df.drop(' funding_total_usd ',axis=1,inplace=True)
    df.drop('country_code',axis=1,inplace=True)
    df.drop('homepage_url',axis=1,inplace=True)
    df.drop('name',axis=1,inplace=True)
#     df['status'] = df['status'].apply(lambda x: x.replace('operating','0')).apply(lambda x: x.replace('acquired','0')).apply(lambda x: x.replace('closed','1'))
#     df['status'] = df['status'].astype('int64')
    return df

def feature_engineer(df):
    '''
    Parameters
        df: Takes in a pandas data frame
    Returns
        a data frame with engineered features
    '''
    df['time_to_funding'] = abs((df['first_funding_at']-df['founded_at']).dt.days)
    df.dropna(inplace=True)
    return df

def create_pie_charts(df,column,column_val,target):
      '''
    Parameters
        df: Cleaned data frame
        column: column of data frame used to split data as string
        column_val: Value we are looking for in column as string
        target: The target values we are trying to predict
    Returns
        a saved image in the images folder
    '''
    column_val_title = ' '.join(re.findall('[A-Z][^A-Z]*',column_val))
    pie_df = df[df[column]==column_val][target].value_counts().rename_axis(target).reset_index(name='counts')
    pie_df['pct'] = pie_df['counts']/len(pie_df)
    labels=pie_df[target]
    fig, ax = plt.subplots(figsize=(14,7))
    ax.pie(pie_df['pct'], explode=[0,0,.15], labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=50)
    ax.axis('equal')
    ax.set_title(f'{target.capitalize()} Of {column_val_title} Market')
    plt.savefig(f'../images/{column_val}_pie.png',dpi=500)


if __name__ == '__main__':
    cleaned_df =clean('../../../Downloads/investments_VC.csv')

    #plot 1
    time_df = cleaned_df['founded_year'].value_counts().rename_axis('year').reset_index(name='counts')
    time_df.sort_values(by='year',inplace=True)
    time_df
    x1 = time_df[time_df['year']>1980]['year']
    y1 = time_df[time_df['year']>1980]['counts']

    funding_df = cleaned_df['first_funding_at'].dt.year.value_counts().rename_axis('year').reset_index(name='counts')
    funding_df['year'] = funding_df['year'].astype('int64')
    funding_df.sort_values(by='year',inplace=True)
    x2 = funding_df[funding_df['year']>1980]['year']
    y2 = funding_df[funding_df['year']>1980]['counts']

    fig,ax = plt.subplots(figsize=(16,8))
    ax.plot(x1,y1,label='Businesses Founded')
    ax.plot(x2,y2,label='First Round Funding Deals')
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('counts')
    ax.set_title('Businesses Founded & First Round Deals',fontweight='bold')
    ax.set_xticks(np.arange(1980, 2016, step=5))
    plt.xticks(rotation=45,ha='center')

    #plot 2
    state_df = cleaned_df['state_code'].value_counts().rename_axis('state').reset_index(name='counts')
    state_df.sort_values(by='state',inplace=True)
    url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
    state_geo = f'{url}/us-states.json'

    m = folium.Map(location=[48, -102], zoom_start=3)

    folium.Choropleth(
        geo_data=state_geo,
        name='All States',
        data=state_df,
        columns=['state', 'counts'],
        key_on='feature.id',
        fill_color='YlGn',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Businesses By State'
    ).add_to(m)

    folium.Choropleth(
        geo_data=state_geo,
        name='Remove CA',
        data=state_df.drop(state_df[state_df['state']=='CA'].index),
        columns=['state', 'counts'],
        key_on='feature.id',
        fill_color='BuPu',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Businesses By State (No CA)'
    ).add_to(m)

    folium.LayerControl().add_to(m)

    m

    #Plot three
    pie_df= cleaned_df['status'].value_counts().rename_axis('status').reset_index(name='counts')
    pie_df['pct'] = pie_df['counts']/len(df)
    labels=pie_df['status']
    fig, ax = plt.subplots(figsize=(14,7))
    ax.pie(pie_df['pct'], explode=[0,0,.15], labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=50)
    ax.axis('equal') 
    ax.set_title('Status')