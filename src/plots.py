import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')
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
    ax.bar(status_df['status'][:1],status_df['pct'][:1])
    ax.bar(status_df['status'][1:2],status_df['pct'][1:2])
    ax.bar(status_df['status'][2:3],status_df['pct'][2:3])
    ax.set_xlabel('Business Status')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Business Status Percentages',fontweight='bold')
    plt.tight_layout()
    plt.savefig('../images/business_status.png')

    #Plot status by market splits
    data = []
    marketlist = ['HealthandWellness','Security','Semiconductors']
    for val in marketlist:
        markets = clean_feat_df[clean_feat_df['market']==val]['status'].value_counts().rename_axis('status').reset_index(name='counts')
        markets['pct'] = round((markets['counts']/markets['counts'].sum())*100,2)
        data.append(markets['pct'].values.tolist())

    operating = [data[0][0],data[1][0],data[2][0]]
    acquired = [data[0][1],data[1][1],data[2][1]]
    closed = [data[0][2],data[1][2],data[2][2]]
    
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)
    width = 0.25
    N = 3
    ind = np.arange(N)
    rects1 = ax.bar(ind, operating, width)
    rects2 = ax.bar(ind+width, acquired, width)
    rects3 = ax.bar(ind+width+width, closed, width)
    ax.set_ylabel('Percentage(%)')
    ax.set_xlabel('Markets')
    ax.set_title('Business Status by Market Sector',fontweight='bold')
    ax.set_xticks(ind + width)
    ax.set_xticklabels( ('Health and Wellness', 'Security', 'Semiconductors') )
    ax.set_yticks(np.arange(0,101,10))
    ax.legend( (rects1[0], rects2[0],rects3[0]), ('Operating', 'Acquired','Closed') )
    plt.tight_layout()
    plt.savefig('../images/market_status.png')

    #plot market sector percentages
    market_split_df = clean_feat_df['market'].value_counts().rename_axis('market').reset_index(name='counts')[1:11]
    market_split_df['pct']=round((market_split_df['counts']/market_split_df['counts'].sum())*100,2)

    fig, ax = plt.subplots(figsize=(14,7))
    ax.bar(market_split_df['market'],market_split_df['pct'])
    ax.set_xlabel('Market')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Top Ten Market Sectors',fontweight='bold')
    ax.set_xticklabels(('Software','Biotechnology','Mobile','Curated Web','Enterprise Software','Health Care','E-Commerce','Hardware & Software','Advertising','Health and Wellness'))
    plt.xticks(rotation=30,ha='right')
    plt.tight_layout()
    plt.savefig('../images/market_split.png')

    #plot status by two different funding types
    fund_data=[]
    funding_df = clean_feat_df[clean_feat_df['venture']>0]['status'].value_counts().rename_axis('status').reset_index(name='counts')
    funding_df['pct'] = round((funding_df['counts']/funding_df['counts'].sum())*100,2)
    fund_data.append(funding_df['pct'].values.tolist())
    funding_df = clean_feat_df[clean_feat_df['equity_crowdfunding']>0]['status'].value_counts().rename_axis('status').reset_index(name='counts')
    funding_df['pct'] = round((funding_df['counts']/funding_df['counts'].sum())*100,2)
    equity_crowd=funding_df['pct'].values.tolist()
    equity_crowd.append(0)
    fund_data.append(equity_crowd)

    N = 2
    operating = [fund_data[0][0],fund_data[1][0]]
    acquired = [fund_data[0][1],fund_data[1][1]]
    closed = [fund_data[0][2],fund_data[1][2]]
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25       # the width of the bars
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, operating, width)
    rects2 = ax.bar(ind+width, acquired, width)
    rects3 = ax.bar(ind+width+width, closed, width)
    ax.set_ylabel('Percentage(%)')
    ax.set_xlabel('Markets')
    ax.set_title('Business Status by Funding Types',fontweight='bold')
    ax.set_xticks(ind + width)
    ax.set_xticklabels( ('Venture', 'Equity Crowd Funding') )
    ax.set_yticks(np.arange(0,101,10))
    ax.legend( (rects1[0], rects2[0],rects3[0]), ('Operating', 'Acquired','Closed') )
    plt.savefig('../images/funding_splits.png')
            