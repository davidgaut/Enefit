# %% [code] {"execution":{"iopub.status.busy":"2024-01-10T08:55:45.315918Z","iopub.execute_input":"2024-01-10T08:55:45.316741Z","iopub.status.idle":"2024-01-10T08:55:47.230037Z","shell.execute_reply.started":"2024-01-10T08:55:45.316701Z","shell.execute_reply":"2024-01-10T08:55:47.228375Z"}}
import sys
import numpy as np 
import polars as pl
import polars.selectors as cs
from sklearn.model_selection import train_test_split
import os
import holidays
holidays_est = holidays.EE()

class Dataset():
    def __init__(self):
        self.IS_KAGGLE = not os.path.isfile("/home/davidgauthier/Codes/Hackatons_2024/Enefit/data/train.csv")
        if self.IS_KAGGLE:
            self.path = '/kaggle/input/predict-energy-behavior-of-prosumers'
        else:
            self.path = "/home/davidgauthier/Codes/Hackatons_2024/Enefit/data"

    def load(self):
        # Data
        self.info      = pl.read_csv(self.path+'/train.csv', try_parse_dates=True).drop_nulls('target')
        self.client    = pl.read_csv(self.path+'/client.csv', try_parse_dates=True)
        self.e_price   = pl.read_csv(self.path+'/electricity_prices.csv', try_parse_dates=True)
        self.g_price   = pl.read_csv(self.path+'/gas_prices.csv', try_parse_dates=True)
        self.f_weather = pl.read_csv(self.path+'/forecast_weather.csv', try_parse_dates=True)
        self.h_weather = pl.read_csv(self.path+'/historical_weather.csv', try_parse_dates=True)
        self.loc_stats = pl.read_csv(self.path+'/loc_stations.csv').with_columns(pl.col('county').cast(pl.Int64))
        self.ws_county = pl.read_csv(self.path+'/weather_station_to_county_mapping.csv', try_parse_dates=True)
        self.county_pos = self.ws_county.drop_nulls().sort('county_name')\
                .group_by(pl.col('county')).agg(
                        pl.col('longitude').map_elements(lambda x: [min(x),max(x)]),
                        pl.col('latitude').map_elements(lambda x: [min(x),max(x)]))

        # Splits
        self.blocks = self.info['data_block_id'].unique().to_numpy()
        self.train_id, self.test_id  = train_test_split(self.blocks, shuffle=False, test_size=0.3)


    def make_features(self,info,client,g_price,e_price,f_weather,h_weather,loc_stats):
        to_drop = ['data_block_id','row_id','prediction_unit_id','county_name','origin_datetime','origin_date']

        info      = info.drop(columns=to_drop)
        client    = client.drop(columns=to_drop)
        g_price   = g_price.drop(columns=to_drop).rename({'forecast_date':'date'})
        e_price   = e_price.drop(columns=to_drop).rename({'forecast_date':'datetime'})
        f_weather = f_weather.drop(columns=to_drop).rename({'forecast_datetime':'datetime'})
        h_weather = h_weather.drop(columns=to_drop)
        loc_stats = loc_stats.drop(columns=to_drop)

        f_weather       = f_weather.filter(pl.col('hours_ahead')>=24).join(loc_stats,on=['latitude','longitude']).drop(columns=['latitude','longitude'])
        h_weather       = h_weather.join(loc_stats,on=['latitude','longitude']).drop(columns=['latitude','longitude'])
        f_weather_date  = f_weather.group_by(['datetime']).mean().drop(columns='county')
        f_weather_local = f_weather.group_by(['datetime','county']).mean()
        h_weather_date  = h_weather.group_by(['datetime']).mean().drop(columns='county')
        h_weather_local = h_weather.group_by(['datetime','county']).mean()

        # Align client date
        client = client.with_columns((pl.col('date') + pl.duration(days=2)).cast(pl.Date))

        target = info.select(pl.col(['target','product_type','datetime','county','is_business','is_consumption']),)
        # Normalize by capacity
        info = info.with_columns(target = target.with_columns(pl.col('datetime').cast(pl.Date).alias('date')).join(client,how='left',on=['date','is_business','county','product_type']).with_columns(pl.col('target')/pl.col('installed_capacity'))['target'])

        #
        target_sum_type = target.drop(columns='product_type').group_by(['datetime','county','is_consumption','is_business']).sum()

        info = info\
            .with_columns(pl.col('datetime').cast(pl.Date).alias('date'))\
            .join(target.with_columns(pl.col('datetime') + pl.duration(days=2)) .rename({'target':'target_1'}),  on=['product_type','datetime','county','is_business','is_consumption'], how='left')\
            .join(target.with_columns(pl.col('datetime') + pl.duration(days=3)) .rename({'target':'target_2'}),  on=['product_type','datetime','county','is_business','is_consumption'], how='left')\
            .join(target.with_columns(pl.col('datetime') + pl.duration(days=4)) .rename({'target':'target_3'}),  on=['product_type','datetime','county','is_business','is_consumption'], how='left')\
            .join(target.with_columns(pl.col('datetime') + pl.duration(days=5)) .rename({'target':'target_4'}),  on=['product_type','datetime','county','is_business','is_consumption'], how='left')\
            .join(target.with_columns(pl.col('datetime') + pl.duration(days=6)) .rename({'target':'target_5'}),  on=['product_type','datetime','county','is_business','is_consumption'], how='left')\
            .join(target.with_columns(pl.col('datetime') + pl.duration(days=7)) .rename({'target':'target_6'}),  on=['product_type','datetime','county','is_business','is_consumption'], how='left')\
            .join(target.with_columns(pl.col('datetime') + pl.duration(days=14)).rename({'target':'target_7'}),  on=['product_type','datetime','county','is_business','is_consumption'],  how='left')\
            .join(target_sum_type.with_columns(pl.col('datetime') + pl.duration(days=2)) .rename({'target':'target_1'}),  on=['datetime','county','is_business','is_consumption' ], suffix='_sum_contract', how='left')\
            .join(target_sum_type.with_columns(pl.col('datetime') + pl.duration(days=3)) .rename({'target':'target_2'}),  on=['datetime','county','is_business','is_consumption' ], suffix='_sum_contract', how='left')\
            .join(target_sum_type.with_columns(pl.col('datetime') + pl.duration(days=7)) .rename({'target':'target_6'}),  on=['datetime','county','is_business','is_consumption' ], suffix='_sum_contract', how='left')\
            .join(client, on=['product_type','county','is_business','date']     ,how='left')\
            .join(e_price.with_columns((pl.col('datetime') + pl.duration(days=1))), on=['datetime'],suffix='_e',how='left')\
            .join(g_price.with_columns((pl.col('date')     + pl.duration(days=1))), on=['date'],    suffix='_g',how='left')\
            .join(f_weather_local,on=['datetime','county'], suffix='_f_l', how='left')\
            .join(f_weather_date .with_columns(pl.col('datetime')+pl.duration(days=2)),on=['datetime'],         suffix='_f_d_2',how='left')\
            .join(h_weather_date .with_columns(pl.col('datetime')+pl.duration(days=2)),on=['datetime'],         suffix='_h_d_2',how='left')\
            .join(f_weather_local.with_columns(pl.col('datetime')+pl.duration(days=2)),on=['datetime','county'],suffix='_f_l_2',  how='left')\
            .join(h_weather_local.with_columns(pl.col('datetime')+pl.duration(days=2)),on=['datetime','county'],suffix='_h_l_2',  how='left')\
            .join(f_weather_date .with_columns(pl.col('datetime')+pl.duration(days=7)),on=['datetime'],         suffix='_f_d_7',how='left')\
            .join(h_weather_date .with_columns(pl.col('datetime')+pl.duration(days=7)),on=['datetime'],         suffix='_h_d_7',how='left')\
            .join(f_weather_local.with_columns(pl.col('datetime')+pl.duration(days=7)),on=['datetime','county'],suffix='_f_l_7',  how='left')\
            .join(h_weather_local.with_columns(pl.col('datetime')+pl.duration(days=7)),on=['datetime','county'],suffix='_h_l_7',  how='left')\
            .with_columns(
                pl.col('datetime').dt.hour()       .alias('hour'),
                pl.col('datetime').dt.day()        .alias('day'),
                pl.col('datetime').dt.ordinal_day().alias('dayofyear'),
                pl.col('datetime').dt.weekday()    .alias('weekday'),
                pl.col('datetime').dt.month()      .alias('month'),
                pl.col('datetime').dt.year()       .alias('year'),
            )\
            .with_columns(
                (pl.col('hour') * 2 * np.pi /  24).sin().alias('sin(hour)'),
                (pl.col('hour') * 2 * np.pi /  24).cos().alias('cos(hour)'),
                (pl.col('year') * 2 * np.pi / 366).sin().alias('sin(year)'),
                (pl.col('year') * 2 * np.pi / 366).cos().alias('cos(year)'),
            )\
            .with_columns(pl.concat_str('county','is_business','product_type','is_consumption',separator='_').alias('segment'))\

        dates = info[['date','datetime']]
        info = info.drop(columns=['date','datetime']).drop_nulls('target')
        info = info.with_columns(target_mean = info[[f"target_{i}" for i in range(1, 7)]].mean_horizontal())\
                .with_columns(target_ratio = info["target_6"] / (info["target_7"] + 1e-3))\
                .with_columns(target_std = (info[[f"target_{i}" for i in range(1, 7)]] - info[[f"target_{i}" for i in range(1, 7)]].mean_horizontal()).with_columns((pl.all()).pow(2)/(len(range(1, 7))-1)).sum_horizontal().pow(1/2).alias('target_std'))
        
        info = info\
                .with_columns(cs.float().cast(pl.Float32()))\
                .with_columns(cs.integer().cast(pl.Int32()))
        return info, dates

    def time_features(self):
        info = self.info.with_columns(pl.col('datetime').dt.weekday().alias('weekday'), pl.col('datetime').dt.month().alias('month'), pl.col('datetime').dt.hour().alias('hour'))

        holidays_ = self.info.select(pl.col('datetime'), pl.col('datetime').dt.strftime('%Y-%m-%d').alias('is_holidays')).group_by('is_holidays').first()
        holidays_est.get('2014-01-01')

        holidays_.with_columns(pl.col('is_holidays').map_elements(lambda x: holidays_est.get(x)))

        holidays_ = holidays_.with_columns(pl.col('is_holidays').is_in([x.strftime('%Y-%m-%d') for x in holidays_est])).filter(pl.col('is_holidays')==True)[['datetime']].sort('datetime').upsample(time_column='datetime',every='1h').with_columns(is_holidays=True)

        info = info.join(holidays_,on='datetime',how='left').with_columns(pl.col('is_holidays').fill_null(False).cast(pl.Int64))

        return info

    def add_features(self):
        pass

    def pl_train_test_split(self,df):    
        return df.filter(pl.col('data_block_id').is_in(self.train_id)),\
                df.filter(pl.col('data_block_id').is_in(self.valid_id)),\
                 df.filter(pl.col('data_block_id').is_in(self.test_id))
    
    def meta_(self):
        self.info.select(~cs.matches('.*_id')).describe()

    def get_county_from_map(self, lat, lon, county_pos) -> pl.Expr:
        return pl.when(pl.col('latitude').is_between(*lat) & pl.col('longitude').is_between(*lon)).then(county_pos).otherwise(-1).alias('county_in_'+str(county_pos))

    def loc_county(self, weather):
        '''Find Estonian county based on coordinates'''
        return weather.with_columns(weather.with_columns([self.get_county_from_map(lat, lon, county) for county, lon, lat in self.county_pos.iter_rows()]).select(cs.matches('county_in_\d*')).max_horizontal().alias('county').cast(pl.Int64)).filter(~(pl.col('county')==-1))

    def fill_weather(self, weather): 
        return pl.concat((weather,weather.filter(pl.col('county')==10).with_columns(county=pl.lit(1,dtype=pl.Int64)),weather.filter(pl.col('county')==11).with_columns(county=pl.lit(8,dtype=pl.Int64))))

    def merge_set(self):

        # Get time features
        info = self.time_features()

        # Production Set
        # Capacity production (prices)
        production = info
        # Add County and fill missing with closest region
        h_weather = self.loc_county(self.h_weather)
        h_weather = self.fill_weather(h_weather)
        f_weather = self.loc_county(self.f_weather)
        f_weather = self.fill_weather(f_weather)
        # Merge production data and weather data
        production = self.production_meteo_(f_weather,production)
        # FB Fill for missing weather forecasts
        production = self.fill_missings(production)
        # Forward contract date
        e_price = self.e_price.with_columns((pl.col('forecast_date')+pl.duration(days=1)).alias('conso_date'))
        g_price = self.g_price.with_columns((pl.col('forecast_date')+pl.duration(days=1)).alias('conso_date'))

        g_price = pl.concat((g_price,g_price[-1].with_columns(cs.matches('.*_date')+pl.duration(days=1))))

        #
        g_price = g_price.with_columns(pl.col('conso_date').cast(pl.Datetime)).sort('conso_date').upsample(time_column='conso_date',every='1h').fill_null(strategy="forward")[:-1]

        # Consumption Set
        production = production.join(e_price,how='left',right_on=['conso_date','data_block_id'],left_on=['datetime','data_block_id'])
        production = production.drop(columns=['origin_date','forecast_date'])

        production = production.join(g_price,how='left',right_on=['conso_date','data_block_id'],left_on=['datetime','data_block_id'])

        production = production.drop(columns=['origin_date','forecast_date'])

        # Add client data
        client = self.client.sort('date').with_columns(pl.col('date').cast(pl.Datetime())).upsample(time_column='date',every='1h').fill_null(strategy='forward')
        production = production.join(client, left_on=['datetime','product_type','county','is_business'],right_on=['date','product_type','county','is_business'], how='left')

        return production

    def production_meteo_(self, f_weather,production):
        # Subset Weather
        f_weather = f_weather.filter(pl.col('hours_ahead').is_between(0,25)).sort(['county','forecast_datetime'])
        # Average across counties !MEAN!
        f_weather = f_weather.group_by(['forecast_datetime','county']).agg(cs.float().mean(),cs.integer().first()).sort(['data_block_id','forecast_datetime'])
        # Merge production + weather forecasts
        return production.join(f_weather, how='left',  left_on=['county','datetime','data_block_id'], right_on=['county','forecast_datetime','data_block_id'])

    def fill_missings(self, production_meteo):
        # Take last meteo forecast
        production_meteo = production_meteo.group_by('county').map_groups(lambda x: x.with_columns(pl.all().forward_fill()).with_columns(pl.all().backward_fill()))
        return production_meteo
    
    def update_data(self,data_block_id,df_new_client,df_new_gas_prices,df_new_electricity_prices,df_new_forecast_weather,df_new_historical_weather,df_new_target):

        df_new_client = pl.from_pandas(
            df_new_client.assign(data_block_id=data_block_id)[self.client.columns], schema_overrides=self.client.schema
        )
        df_new_gas_prices = pl.from_pandas(
            df_new_gas_prices.assign(data_block_id=data_block_id)[self.g_price.columns], schema_overrides=self.g_price.schema
        )
        df_new_electricity_prices = pl.from_pandas(
            df_new_electricity_prices.assign(data_block_id=data_block_id)[self.e_price.columns], schema_overrides=self.e_price.schema
        )
        df_new_forecast_weather = pl.from_pandas(
            df_new_forecast_weather.assign(data_block_id=data_block_id)[self.f_weather.columns], schema_overrides=self.f_weather.schema
        )
        df_new_historical_weather = pl.from_pandas(
            df_new_historical_weather.assign(data_block_id=data_block_id)[self.h_weather.columns], schema_overrides=self.h_weather.schema
        )
        df_new_target = pl.from_pandas(
            df_new_target.assign(data_block_id=data_block_id)[self.info.columns], schema_overrides=self.info.schema
        )

        self.client    = pl.concat((self.client,df_new_client))
        self.g_price   = pl.concat((self.g_price,df_new_gas_prices))
        self.e_price   = pl.concat((self.e_price,df_new_electricity_prices))
        self.f_weather = pl.concat((self.f_weather,df_new_forecast_weather))
        self.h_weather = pl.concat((self.h_weather,df_new_historical_weather))
        self.info      = pl.concat((self.info,df_new_target))


# def cv_(sets_):
#     ids_ = sets_.with_row_count()
#     yield ids_.filter(pl.col('data_block_id').is_in(dataset.train_id))['row_nr'].to_numpy(),\
#         ids_.filter(pl.col('data_block_id').is_in(dataset.test_id))['row_nr'].to_numpy()