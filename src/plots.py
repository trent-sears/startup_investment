import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17})
import folium
import re
from models import feature_engineer, DataFrame

if __name__ == '__main__':

    #intialized dataframe class and build features
    intial_df = DataFrame('../../../Downloads/investments_VC.csv').clean()
    clean_feat_df=feature_engineer(intial_df)

    #plot businesses opened versus first round deals
    time_df = clean_feat_df['founded_year'].value_counts()\
        .rename_axis('year').reset_index(name='counts')
    time_df.sort_values(by='year',inplace=True)
    time_df
    x1 = time_df[time_df['year']>1980]['year']
    y1 = time_df[time_df['year']>1980]['counts']

    funding_df = clean_feat_df['first_funding_at'].dt.year.value_counts()\
        .rename_axis('year').reset_index(name='counts')
    funding_df['year'] = funding_df['year'].astype('int64')
    funding_df.sort_values(by='year',inplace=True)
    x2 = funding_df[funding_df['year']>1980]['year']
    y2 = funding_df[funding_df['year']>1980]['counts']

    fig,ax = plt.subplots(figsize=(14,7))
    ax.plot(x1,y1,label='Businesses Founded')
    ax.plot(x2,y2,label='First Round Funding Deals')
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('counts')
    ax.set_title('Businesses Founded & First Round Deals',fontweight='bold')
    ax.set_xticks(np.arange(1980, 2016, step=5))
    plt.xticks(rotation=45,ha='center')
    plt.tight_layout()
    plt.savefig('../images/founded_vs_deals.png',dpi=500)


    #plot using folium to look at business and status by state
    state_df = clean_feat_df['state_code'].value_counts()\
        .rename_axis('state').reset_index(name='counts')
    state_df.sort_values(by='state',inplace=True)
    acquired_df = clean_feat_df[clean_feat_df['status']=='acquired']['state_code']\
        .value_counts().rename_axis('state').reset_index(name='counts')
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

    folium.Choropleth(
        geo_data=state_geo,
        name='Acquired Businesses',
        data=acquired_df,
        columns=['state', 'counts'],
        key_on='feature.id',
        fill_color='OrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Acquired Businesses'
    ).add_to(m)

    folium.Choropleth(
        geo_data=state_geo,
        name='acquired Businesses remove CA',
        data=acquired_df.drop(acquired_df[acquired_df['state']=='CA'].index),
        columns=['state', 'counts'],
        key_on='feature.id',
        fill_color='OrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='acquired Businesses (No CA)'
    ).add_to(m)
    folium.LayerControl().add_to(m)
    m.save('../images/business_map.html')
    
    #Plot status for entire populattion
    status_df = clean_feat_df['status'].value_counts().rename_axis('status').reset_index(name='counts')
    status_df ['pct'] = round((status_df['counts']/status_df['counts'].sum())*100,2)
    status_df
    labels=status_df['status']
    fig, ax = plt.subplots(figsize=(14,7))
    ax.bar(status_df['status'],status_df['pct'])
    ax.set_xlabel('Business Status')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Business Status Percentages',fontweight='bold')
    plt.tight_layout()
    plt.savefig('../images/business_status.png')
        