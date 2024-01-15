# https://www.kaggle.com/code/jeanbaptistescellier/points-where-weather-is-given

import polars as pl
import requests
from itertools import chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import contextily as ctx

forecast_weather_df = pd.read_csv('./data/forecast_weather.csv')
ws_county = pd.read_csv('./data/weather_station_to_county_mapping.csv')
unique_coords_forecast = np.unique(forecast_weather_df[["latitude", "longitude"]], axis=0)


points = [Point(x, y) for y, x in unique_coords_forecast]
points_gdf = gpd.GeoDataFrame(points, columns=['geometry'])

def order_ways(list_list_points):
    #list_first_points = [list_points[0] for list_points in list_list_points]
    #list_last_points = [list_points[-1] for list_points in list_list_points]
    ordered_list_list_points = []
    first_list_point = list_list_points[0]
    list_points = list_list_points[0]
    last_point = 0
    while (last_point != first_list_point[-1]):
        ordered_list_list_points.append(list_points)
        list_list_points.remove(list_points)
        list_first_points = [list_points[0] for list_points in list_list_points]
        list_last_points = [list_points[-1] for list_points in list_list_points]
        #print(list_points[0])
        #print(list_points[-1])
        #print()
        last_point = list_points[-1]
        try:
            index_list_points = list_first_points.index(last_point)
            list_points = list_list_points[index_list_points]
        except ValueError:
            try:
                index_list_points = list_last_points.index(last_point)
                list_points = list_list_points[index_list_points]
                list_points.reverse()
            except ValueError:
                break
        last_point = list_points[-1]
    return ordered_list_list_points

##
# create query
overpass_query_counties = """
[out:json];
area["name:en"="Estonia"]->.searchArea;
(
  relation["admin_level"="6"](area.searchArea);
);
out geom;
"""


# get Estonia boundaries from overpass
response = requests.post("https://overpass-api.de/api/interpreter", data=overpass_query_counties)
estonia_geojson = response.json()

# parse geometry
geometry = []
names = []
for element in estonia_geojson['elements']:
    members = element['members']
    name = element["tags"]["alt_name"]
    names.append(name)
    coords_poly = []
    for member in members:
        if member['type'] == 'way' and 'geometry' in member:
            coords = [(node['lon'], node['lat']) for node in member['geometry']]
            coords_poly.append(coords)
            #geometry.append(LineString(coords))
    coords_poly = order_ways(coords_poly)
    coords_poly = list(chain(*coords_poly))
    geometry.append(Polygon(coords_poly))

name_series = pd.Series(names, name="County")
gdf = gpd.GeoDataFrame(name_series, geometry=geometry)
gdf = gdf.set_index("County")
gdf.crs = 'EPSG:4326'
# create query
overpass_query_land_area = """
[out:json];
area["name:en"="Estonia"]->.searchArea;
(
  relation[boundary=land_area][admin_level=2](area.searchArea);
);
out geom;
"""

# get Estonia boundaries from overpass
response = requests.post("https://overpass-api.de/api/interpreter", data=overpass_query_land_area)
land_area_geojson = response.json()

# parse geometry
geometry = []
members = land_area_geojson['elements'][0]['members']
coords_poly = []
for member in members:
    if member['type'] == 'way' and 'geometry' in member:
        coords = [(node['lon'], node['lat']) for node in member['geometry']]
        coords_poly.append(coords)
        #geometry.append(LineString(coords))
coords_poly = order_ways(coords_poly)
coords_poly = list(chain(*coords_poly))
geometry.append(Polygon(coords_poly))

gdf_land = gpd.GeoDataFrame(geometry=geometry)
gdf_land.crs = 'EPSG:4326'
# create query
overpass_query_hiiumaa = """
[out:json];
area["name:en"="Estonia"]->.searchArea;
(
  relation[place=island][name="Hiiumaa"](area.searchArea);
);
out geom;
"""

# get Estonia boundaries from overpass
response = requests.post("https://overpass-api.de/api/interpreter", data=overpass_query_hiiumaa)
hiiumaa_geojson = response.json()

# parse geometry
geometry = []
members = hiiumaa_geojson['elements'][0]['members']
coords_poly = []
for member in members:
    if member['type'] == 'way' and 'geometry' in member:
        coords = [(node['lon'], node['lat']) for node in member['geometry']]
        coords_poly.append(coords)
        #geometry.append(LineString(coords))
coords_poly = order_ways(coords_poly)
coords_poly = list(chain(*coords_poly))
geometry.append(Polygon(coords_poly))

gdf_hiiumaa = gpd.GeoDataFrame(geometry=geometry)
gdf_hiiumaa.crs = 'EPSG:4326'
# create query
overpass_query_saaremaa = """
[out:json];
area["name:en"="Estonia"]->.searchArea;
(
  relation[place=island][name="Saaremaa"](area.searchArea);
);
out geom;
"""

# get Estonia boundaries from overpass
response = requests.post("https://overpass-api.de/api/interpreter", data=overpass_query_saaremaa)
saaremaa_geojson = response.json()

# parse geometry
geometry = []
members = saaremaa_geojson['elements'][0]['members']
coords_poly = []
for member in members:
    if member['type'] == 'way' and 'geometry' in member:
        coords = [(node['lon'], node['lat']) for node in member['geometry']]
        coords_poly.append(coords)
        #geometry.append(LineString(coords))
coords_poly = order_ways(coords_poly)
coords_poly = list(chain(*coords_poly))
geometry.append(Polygon(coords_poly))

gdf_saaremaa = gpd.GeoDataFrame(geometry=geometry)
gdf_saaremaa.crs = 'EPSG:4326'

# %%
fig, ax = plt.subplots(figsize=(18, 18))
gdf_land.plot(ax=ax, color='blue', linewidth=1)
gdf_hiiumaa.plot(ax=ax, color='green', linewidth=1)
gdf_saaremaa.plot(ax=ax, color='orange', linewidth=1)
plt.show()
# %%
for county in gdf.index:
    points_gdf[county] = points_gdf.apply(lambda x: x.geometry.within(gdf.loc[county, "geometry"]), axis=1)

points_gdf["county"] = points_gdf[gdf.index].apply(lambda x: x[x].index[0] if len(x[x]) > 0 else False, axis=1)
points_gdf["land"] = points_gdf.apply(lambda x: x.geometry.within(gdf_land.loc[0, "geometry"]), axis=1)
points_gdf["saaremaa"] = points_gdf.apply(lambda x: x.geometry.within(gdf_saaremaa.loc[0, "geometry"]), axis=1)
points_gdf["hiiumaa"] = points_gdf.apply(lambda x: x.geometry.within(gdf_hiiumaa.loc[0, "geometry"]), axis=1)
points_gdf["land"] = points_gdf["land"] | points_gdf["saaremaa"] | points_gdf["hiiumaa"]
points_gdf["county"] = points_gdf["county"] * points_gdf["land"]
points_gdf["county"] = points_gdf["county"].replace(["", 0], np.nan)
points_gdf = points_gdf[["geometry", "county"]]

points_gdf["latitude"] = points_gdf.geometry.apply(lambda x: x.y)
points_gdf["longitude"] = points_gdf.geometry.apply(lambda x: x.x)

# %%
fig, ax = plt.subplots(figsize=(13, 8))
gdf.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=1)
sns.scatterplot(data=points_gdf, x='longitude', y='latitude', hue='county')
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_aspect('equal')
ax.set_title('Counties land location points')
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

#%%
#
forecast_weather_df_county = forecast_weather_df.merge(points_gdf.dropna().drop(columns=['geometry']),on=['latitude','longitude'],how='left')

forecast_weather_df_county = forecast_weather_df_county.dropna(subset='county').drop_duplicates()

loc_stations = forecast_weather_df_county[['latitude','longitude','county']].rename(columns={'county':'county_name'}).merge(ws_county[['county_name','county']].drop_duplicates().dropna(),on=['county_name']).drop_duplicates()
loc_stations.reset_index(drop=True).to_csv('./data/loc_stations.csv',index=None)

# %%
# Simply merging on lat and long
forecast_weather_df_county_lat_long = forecast_weather_df.merge(ws_county,on=['latitude','longitude'],how='left')

forecast_weather_df_county_lat_long = forecast_weather_df_county_lat_long.drop_duplicates().dropna(subset='county')

# %%
fig, ax = plt.subplots(figsize=(13, 8))
gdf.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=1)
sns.scatterplot(data=forecast_weather_df_county_lat_long, x='longitude', y='latitude', hue='county_name')
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_aspect('equal')
ax.set_title('Counties land location points')
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

# %%
fig, ax = plt.subplots(figsize=(13, 8))
gdf.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=1)
sns.scatterplot(data=forecast_weather_df_county, x='longitude', y='latitude', hue='county')
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_aspect('equal')
ax.set_title('Counties land location points')
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))


# %%
# Segment and locations
info = pl.read_csv("/home/davidgauthier/Codes/Hackatons_2024/Enefit/data"+'/train.csv', try_parse_dates=True)

info = info.filter(pl.col('is_consumption')==0).with_columns(segment=pl.concat_str('county','is_business','product_type',separator='_'))

h_weather = pl.read_csv("/home/davidgauthier/Codes/Hackatons_2024/Enefit/data"+'/historical_weather.csv', try_parse_dates=True)

h_weather = h_weather.join(h_weather[['latitude','longitude']].unique().with_row_count('location'),how='left',on=['latitude','longitude'])

segments = info['segment'].unique().to_numpy()
seg_corrs = list()
locs = h_weather['location'].unique().to_numpy()
target = ['direct_solar_radiation',]
for segment in segments:
    ii = info.filter(pl.col('segment')==segment).join(h_weather[['datetime','location']+target],how='left',on=['datetime']).drop_nulls()
    dict_corr = list()
    for loc in locs:
        corr = ii.filter(pl.col('location')==loc)[target+['target']].corr()[0,1]
        dict_corr.append((loc,corr))
    seg_corrs.append((segment,dict_corr))

seg_corrs = dict(seg_corrs)
corr_frame = pl.concat([pl.DataFrame(seg_corrs[seg]).rename({'column_0':'loc','column_1':seg}) for seg in segments],how='align').with_columns(pl.col('loc').cast(pl.Int16))

corr_frame = corr_frame[sorted(corr_frame.columns)].drop(columns=['loc'])
print(corr_frame.with_columns(pl.all().arg_max())[0])

loc_coor = corr_frame.with_columns(pl.all().arg_max())[0].transpose(include_header=True).rename({'column_0':'location','column':'segment'}).join(h_weather[['location','latitude','longitude']],how='left',on='location').unique()
loc_coor.write_csv('./data/loc_stations_corr.csv',)

# %%
fig, ax = plt.subplots(figsize=(13, 12))
gdf.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=1)
ff = forecast_weather_df.merge(h_weather[['latitude','longitude','location']].unique().to_pandas(),how='left',on=['latitude','longitude'])[['latitude','longitude','location',]].drop_duplicates()
sns.scatterplot(data=ff, x='longitude', y='latitude', )
for line in range(0,ff.shape[0]):
     plt.text(ff.longitude[line]+0.05, ff.latitude[line], ff.location[line], horizontalalignment='left', size='medium', color='black', weight='semibold')

ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_aspect('equal')
ax.set_title('Counties land location points')
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

# %%
info = pl.read_csv("/home/davidgauthier/Codes/Hackatons_2024/Enefit/data"+'/train.csv', try_parse_dates=True)

info = info.filter(pl.col('is_consumption')==1).with_columns(segment=pl.concat_str('county','is_business','product_type',separator='_'))

h_weather = pl.read_csv("/home/davidgauthier/Codes/Hackatons_2024/Enefit/data"+'/historical_weather.csv', try_parse_dates=True)

h_weather = h_weather.join(h_weather[['latitude','longitude']].unique().with_row_count('location'),how='left',on=['latitude','longitude'])

segments = info['segment'].unique().to_numpy()
seg_corrs = list()
locs = h_weather['location'].unique().to_numpy()
target = ['temperature',]
['location']
for segment in segments:
    ii = info.filter(pl.col('segment')==segment).join(h_weather[['datetime','location']+target],how='left',on=['datetime']).drop_nulls()
    dict_corr = list()
    for loc in locs:
        corr = ii.filter(pl.col('location')==loc)[target+['target']].corr()[0,1]
        dict_corr.append((loc,corr))
    seg_corrs.append((segment,dict_corr))

seg_corrs = dict(seg_corrs)
corr_frame = pl.concat([pl.DataFrame(seg_corrs[seg]).rename({'column_0':'loc','column_1':seg}) for seg in segments],how='align').with_columns(pl.col('loc').cast(pl.Int16))

corr_frame = corr_frame[sorted(corr_frame.columns)]
print(corr_frame.with_columns(pl.all().arg_max())[0])

# %%
# Study meteo and powr