# %%
from functions import Dataset

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

import pandas as pd
import polars as pl
import polars.selectors as cs
import random
import os

import numpy as np

from functions import Dataset

plt.style.use('seaborn')
rcParams["figure.figsize"] = [15,10]

# Data
dataset = Dataset()
dataset.load()

# Sets
sets_  =  dataset.merge_set()
# consomation =  dataset.consomation_set()



# %%
info      = dataset.info
client    = dataset.client
e_prices  = dataset.e_price
g_prices  = dataset.g_price
h_weather = dataset.h_weather
f_weather = dataset.f_weather

# %%
# Target Distribution by Types
df = info.with_columns(
    pl.col('target').clip_max(pl.col('target').quantile(0.95))
)
sns.histplot(data=df,x='target',hue='is_consumption')

# Share of zero-production
info.with_columns((pl.col('target')!=0).cast(pl.Int64)).group_by(['is_consumption','is_business']).agg(pl.map_groups(['target'],lambda x: round(100*np.mean(x))))

# Consumption is almost always positive
info.with_columns(pl.col('target')==0).group_by(['is_consumption']).sum()

#
groups_ = info.group_by(['county','is_business','product_type','is_consumption']).agg(pl.col('datetime'),pl.col('target'))

#
sample = info.filter(pl.col('county')==1)
g = sns.FacetGrid(data=sample,row='is_business',col='product_type')
g.map(sns.lineplot, data=sample, x='datetime', y='target', hue='is_consumption')

#
sns.lineplot(data=e_prices.with_columns(pl.col('euros_per_mwh')),x='forecast_date',y='euros_per_mwh')

sns.lineplot(data=g_prices,x='forecast_date',y='lowest_price_per_mwh')
sns.lineplot(data=g_prices,x='forecast_date',y='highest_price_per_mwh')


# %%
sns.lineplot(data=client,x='date',y='installed_capacity',hue='county')

# Number of customer (solar panels)
sns.lineplot(data=info.group_by(['is_business','prediction_unit_id']).agg(pl.col('datetime').first(), pl.col('target').count()).sort('datetime'),x='datetime',y='target',hue='is_business')

# %% [markdown]
#

#-----------------------------------------------------------------------------------


# %%
# Unkown County (12)
unkown_county = production.filter(pl.col('county')==12)
unkown_county.group_by('county').agg(pl.col('product_type').unique(), pl.col('is_business').unique(), pl.col('prediction_unit_id').unique())
sns.histplot(data=unkown_county,x='target',stat='percent',alpha=0.8,color='r',log_scale=True)
sns.histplot(data=production.filter(pl.col('county')==10),x='target',stat='percent',alpha=0.4,log_scale=True)
sns.histplot(data=production.filter(pl.col('county')==13),x='target',stat='percent',alpha=0.4,log_scale=True)

# Hiiumaa (county 1) missing for observed weather
sns.lineplot(data=f_weather.filter(pl.col('county').is_in([1,12,10,0])),x='forecast_datetime',y='temperature',hue='county',errorbar=None)


# %%
# Energy Prices and Capacity

# Solar Panel Capacity in time

# Per county
_, axes = plt.subplots(3,1,sharex=True,sharey=False)
sns.lineplot(data=client.group_by(['county','date']).agg(pl.col('installed_capacity').sum()),x='date',y='installed_capacity',hue='county',ax=axes[0])
axes[0].set_title('Capacity per Segment')

# Per Industry
total_capacity =  client.filter(pl.col('is_business')==1).group_by('date').agg(pl.col('installed_capacity').sum())
axes[1] = sns.lineplot(data=total_capacity,x='date',y='installed_capacity',ax=axes[1])
axes[1].set_title('Total Capacity')
total_capacity =  client.filter(pl.col('is_business')==0).group_by('date').agg(pl.col('installed_capacity').sum())
axes[1] = sns.lineplot(data=total_capacity,x='date',y='installed_capacity',ax=axes[1])

axes[1] = sns.lineplot(data=client.group_by('date').agg(pl.col('installed_capacity').sum()),x='date',y='installed_capacity',ax=axes[1])
axes[1].set_title('Cumulative Solar Capacity')
axes[1].legend(['Business','Households','Total'])

# Electricity
ax_tw = sns.lineplot(data=e_prices.with_columns(pl.col('euros_per_mwh').clip(upper_bound=pl.col('euros_per_mwh').quantile(0.999))),x='forecast_date',y='euros_per_mwh',c='r',alpha=0.4,ax=axes[2])
sns.lineplot(data=e_prices.sort('forecast_date').rolling(index_column='forecast_date',period='1d').agg(pl.col('euros_per_mwh').mean()),x='forecast_date',y='euros_per_mwh',c='r',alpha=0.4,ax=axes[2])
axes[2].grid(False)
axes[2].legend(['electricity','electricity MA (1d)',])

# Gas
twx = axes[2].twinx()
ax_tw = sns.lineplot(data=g_prices,x='forecast_date',y='lowest_price_per_mwh',c='b',alpha=0.4,ax=twx)
sns.lineplot(data=g_prices,x='forecast_date',y='highest_price_per_mwh',c='b',alpha=0.4,ax=twx)
twx.grid(False)
twx.legend(['gas',])


#
# Plots
#
production_meteo.select(cs.float()).drop_nulls(['temperature','target']).corr()
#
sns.lineplot(production_meteo[['target','direct_solar_radiation','datetime']].drop_nulls())

# plt.scatter(x=f_weather['longitude'], y=f_weather['latitude'])
# plt.show()

# import pandas as pd
# from shapely.geometry import Point
# import geopandas as gpd
# from geopandas import GeoDataFrame

# import plotly.express as px

# fig = px.scatter_geo(f_weather ,lat='latitude',lon='longitude', )
# fig.update_layout(title = 'World map', title_x=0.5)
# fig.show()

# geometry = [Point(xy) for xy in zip(f_weather['longitude'], f_weather['latitude'])]
# gdf = GeoDataFrame(df, geometry=geometry)   

# #this is a simple map that goes with geopandas
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);
# %%
# Meteo 



#%%
# Weather and Production

info
#%%
# How much of production can be forecaster from forecasted radiations and capacity

# Join local geo data and production data